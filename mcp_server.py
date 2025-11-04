#mcp_server.py

#!/usr/bin/env python3
"""
Complete MCP Server for Content & Media Processing
Standalone implementation with all services integrated
(*** FINAL STABLE VERSION ***)
"""

import asyncio
import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence
from datetime import datetime
import tempfile
import shutil
import uuid
import sqlite3

# Core MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Processing libraries
from PIL import Image, ImageEnhance
import pypandoc
from docx import Document
import PyPDF2
from bs4 import BeautifulSoup
import markdown
import feedparser
import requests
import email
import mailbox
from email.parser import BytesParser
from email.policy import default
import re
import magic

# <<< LIBRARIES FOR RELIABLE CONVERSION >>>
import pythoncom
import win32com.client

# Configuration
class Config:
    BASE_DIR = Path(__file__).parent
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUT_DIR = BASE_DIR / "output" 
    CACHE_DIR = BASE_DIR / "cache"
    DB_PATH = BASE_DIR / "mcp_server.db"
    
    # File size limits (in bytes)
    MAX_DOC_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_IMG_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_EMAIL_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    
    # Supported formats
    DOC_FORMATS = ["pdf", "docx", "doc", "html", "htm", "md", "markdown", "txt", "rtf", "pptx"]
    IMG_FORMATS = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]
    EMAIL_FORMATS = ["mbox", "eml"]
    
    def __post_init__(self):
        for directory in [self.TEMP_DIR, self.OUTPUT_DIR, self.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

config = Config()
config.__post_init__()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP Server
app = Server("content-media-hub")

# --- Database Utility Functions ---
def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def log_to_history(tool_name: str, status: str, source_path: str = None, output_path: str = None, error_message: str = None, metadata: dict = None):
    """Log an action to the processing_history table"""
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO processing_history (tool_name, source_path, output_path, status, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tool_name, source_path, output_path, status, error_message, json.dumps(metadata) if metadata else None)
            )
    except Exception as e:
        logger.error(f"Failed to log to database: {e}")

# --- Utility Functions ---
def ensure_directory(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def get_file_hash(filepath: str) -> str:
    """Generate hash for file"""
    import hashlib
    return hashlib.md5(filepath.encode()).hexdigest()[:8]

def validate_file_size(filepath: Path, max_size: int):
    """Validate file size"""
    if filepath.stat().st_size > max_size:
        raise ValueError(f"File too large: {filepath.stat().st_size / (1024*1024):.1f}MB")

def detect_file_format(filepath: Path) -> str:
    """Detect file format"""
    ext = filepath.suffix.lower().lstrip('.')
    try:
        mime_type = magic.from_file(str(filepath), mime=True)
        if 'pdf' in mime_type:
            return 'pdf'
        elif 'word' in mime_type or 'officedocument.wordprocessingml' in mime_type:
            return 'docx' if 'openxml' in mime_type else 'doc'
        elif 'powerpoint' in mime_type or 'officedocument.presentationml' in mime_type:
            return 'pptx'
        elif 'html' in mime_type:
            return 'html'
        elif 'image' in mime_type:
            return ext if ext in config.IMG_FORMATS else 'jpg'
    except:
        pass
    return ext if ext in config.DOC_FORMATS + config.IMG_FORMATS else 'txt'

# --- Document Processing Service ---
class DocumentProcessor:
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR / "documents"
        ensure_directory(self.output_dir)
    
    async def convert_document(self, source_path: str, target_format: str, **kwargs) -> Dict[str, Any]:
        """Convert document between formats"""
        try:
            # Use Path() to handle file paths correctly
            source = Path(source_path)
            if not source.exists():
                # Try to find the file in the base directory if a full path fails
                source = config.BASE_DIR / source.name
                if not source.exists():
                    raise FileNotFoundError(f"Source file not found: {source_path}")

            validate_file_size(source, config.MAX_DOC_SIZE)
            source_format = detect_file_format(source)

            # Generate output path
            output_name = f"{source.stem}.{target_format}"
            output_path = self.output_dir / output_name

            # Perform conversion
            if source_format == target_format:
                shutil.copy2(source, output_path)
                message = "File copied (same format)"

            elif (source_format == 'docx' or source_format == 'doc') and target_format == 'pdf':
                await self._com_convert_to_pdf(source, output_path, "Word.Application", "Documents", 17)
                await asyncio.sleep(1.5) # Wait for file lock
                message = "Converted Document to PDF using Word"
            
            elif (source_format == 'pptx' or source_format == 'ppt') and target_format == 'pdf':
                await self._com_convert_to_pdf(source, output_path, "PowerPoint.Application", "Presentations", 32)
                await asyncio.sleep(1.5) # Wait for file lock
                message = "Converted Presentation to PDF using PowerPoint"

            elif self._can_use_pandoc(source_format, target_format):
                await self._pandoc_convert(source, output_path, source_format, target_format)
                message = "Converted using Pandoc"
            elif source_format == 'pdf' and target_format == 'txt':
                await self._pdf_to_text(source, output_path)
                message = "Converted PDF to text"
            elif source_format == 'docx' and target_format == 'txt':
                await self._docx_to_text(source, output_path)
                message = "Converted DOCX to text"
            else:
                raise ValueError(f"Unsupported conversion: from '{source_format}' to '{target_format}'")

            metadata = await self._get_document_metadata(output_path)

            result = {
                "success": True,
                "message": message,
                "output_path": str(output_path),
                "source_format": source_format,
                "target_format": target_format,
                "file_size": output_path.stat().st_size,
                "metadata": metadata
            }
            log_to_history("convert_document", "success", str(source), str(output_path), metadata=result)
            return result

        except Exception as e:
            logger.exception(f"Document conversion failed")
            log_to_history("convert_document", "error", str(source_path), error_message=str(e))
            return {
                "success": False,
                "error": str(e),
                "source_path": str(source_path),
                "target_format": target_format
            }
    
    def _can_use_pandoc(self, source_format: str, target_format: str) -> bool:
        """Check if pandoc can handle conversion"""
        pandoc_sources = {'md', 'markdown', 'html', 'htm', 'docx', 'txt', 'rtf'}
        pandoc_targets = {'md', 'markdown', 'html', 'htm', 'docx', 'txt', 'rtf', 'pdf'} 
        return source_format in pandoc_sources and target_format in pandoc_targets
    
    async def _pandoc_convert(self, source: Path, output: Path, source_fmt: str, target_fmt: str):
        """Convert using pandoc"""
        try:
            extra_args = []
            if target_fmt == 'html':
                extra_args = ['--standalone']
            elif target_fmt == 'pdf':
                extra_args = ['--pdf-engine=wkhtmltopdf']

            pypandoc.convert_file(
                str(source), target_fmt, outputfile=str(output),
                extra_args=extra_args
            )
        except Exception as e:
            if 'wkhtmltopdf' in str(e):
                raise RuntimeError("Pandoc conversion failed: PDF engine 'wkhtmltopdf' not found. Please install it to convert to PDF.")
            raise RuntimeError(f"Pandoc conversion failed: {e}")
    
    async def _pdf_to_text(self, source: Path, output: Path):
        """Convert PDF to text"""
        try:
            with open(source, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_content = []
                for page in reader.pages:
                    text_content.append(page.extract_text())
            
            with open(output, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(text_content))
        except Exception as e:
            raise RuntimeError(f"PDF to text conversion failed: {e}")
    
    async def _docx_to_text(self, source: Path, output: Path):
        """Convert DOCX to text"""
        try:
            doc = Document(str(source))
            paragraphs = [p.text for p in doc.paragraphs]
            
            with open(output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(paragraphs))
        except Exception as e:
            raise RuntimeError(f"DOCX to text conversion failed: {e}")
    
    async def _to_html(self, source: Path, output: Path, source_fmt: str):
        """Convert various formats to HTML"""
        if source_fmt in ['md', 'markdown']:
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
            html_content = markdown.markdown(content)
            with open(output, 'w', encoding='utf-8') as f:
                f.write(f"<html><body>{html_content}</body></html>")
        elif source_fmt == 'txt':
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
            html_content = f"<html><body><pre>{content}</pre></body></html>"
            with open(output, 'w', encoding='utf-8') as f:
                f.write(html_content)
    
    async def _from_html(self, source: Path, output: Path, target_fmt: str):
        """Convert HTML to other formats"""
        if target_fmt == 'txt':
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text()
            with open(output, 'w', encoding='utf-8') as f:
                f.write(text_content)
    
    async def _get_document_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract document metadata"""
        stats = filepath.stat()
        return {
            "file_size": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "format": detect_file_format(filepath)
        }
        
    async def _com_convert_to_pdf(self, source: Path, output: Path, app_name: str, collection_name: str, pdf_format_type: int):
        """Generic COM converter for Word and PowerPoint"""
        
        def com_task():
            pythoncom.CoInitialize()
            app = None
            doc = None
            try:
                src_abs = str(source.resolve())
                out_abs = str(output.resolve())
                
                app = win32com.client.Dispatch(app_name)
                collection = getattr(app, collection_name)
                doc = collection.Open(src_abs, WithWindow=False)
                
                doc.SaveAs(out_abs, pdf_format_type) # 17 for Word, 32 for PowerPoint
                
            except Exception as e:
                raise RuntimeError(f"{app_name} (COM) conversion failed: {e}. Do you have Microsoft Office installed?")
            finally:
                if doc:
                    doc.Close()
                if app:
                    app.Quit()
                pythoncom.CoUninitialize()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, com_task)
        except Exception as e:
            raise RuntimeError(f"COM conversion failed: {e}")

# --- Image Processing Service ---
class ImageProcessor:
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR / "images"
        ensure_directory(self.output_dir)
    
    async def process_image(self, image_path: str, operations: List[str], **kwargs) -> Dict[str, Any]:
        """Process image with specified operations"""
        try:
            source = Path(image_path)
            if not source.exists():
                source = config.BASE_DIR / source.name
                if not source.exists():
                    raise FileNotFoundError(f"Source file not found: {image_path}")
            
            validate_file_size(source, config.MAX_IMG_SIZE)
            
            output_format = kwargs.get('output_format', 'png')
            quality = kwargs.get('quality', 85)
            width = kwargs.get('width')
            height = kwargs.get('height')
            
            # Generate output path
            output_name = f"{source.stem}_processed.{output_format}"
            output_path = self.output_dir / output_name
            
            # Process image
            with Image.open(source) as img:
                processed_img = img.copy()
                
                # Apply operations
                for operation in operations:
                    if operation == "resize" and (width or height):
                        processed_img = self._resize_image(processed_img, width, height)
                    elif operation == "enhance":
                        processed_img = self._enhance_image(processed_img)
                    elif operation == "convert":
                        pass  # Handled in save
                
                # Convert mode if necessary
                if processed_img.mode in ('RGBA', 'LA', 'P') and output_format.lower() == 'jpeg':
                    background = Image.new('RGB', processed_img.size, (255, 255, 255))
                    if processed_img.mode in ('RGBA', 'LA'):
                        background.paste(processed_img, mask=processed_img.split()[-1])
                    else:
                        background.paste(processed_img)
                    processed_img = background
                
                # Save processed image
                save_kwargs = {'format': output_format.upper()}
                if output_format.lower() in ['jpeg', 'webp']:
                    save_kwargs.update({'quality': quality, 'optimize': True})
                
                processed_img.save(output_path, **save_kwargs)
            
            metadata = await self._get_image_metadata(output_path)
            
            result = {
                "success": True,
                "message": f"Image processed with {len(operations)} operations",
                "output_path": str(output_path),
                "operations_applied": operations,
                "output_format": output_format,
                "metadata": metadata
            }
            log_to_history("process_image", "success", str(source), str(output_path), metadata=result)
            return result
            
        except Exception as e:
            logger.exception(f"Image processing failed")
            log_to_history("process_image", "error", str(image_path), error_message=str(e))
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "operations": operations
            }
    
    def _resize_image(self, img: Image.Image, width: Optional[int], height: Optional[int]) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        if not width and not height:
            return img
        
        original_width, original_height = img.size
        
        if width and height:
            new_size = (width, height)
        elif width:
            ratio = width / original_width
            new_size = (width, int(original_height * ratio))
        else:  # height only
            ratio = height / original_height
            new_size = (int(original_width * ratio), height)
        
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Apply basic image enhancements"""
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        return img
    
    async def _get_image_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract image metadata"""
        try:
            with Image.open(filepath) as img:
                stats = filepath.stat()
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": stats.st_size,
                    "has_exif": hasattr(img, '_getexif') and img._getexif() is not None
                }
        except Exception:
            stats = filepath.stat()
            return {"file_size": stats.st_size, "format": "unknown"}

# --- Email Processing Service ---
class EmailProcessor:
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR / "email"
        ensure_directory(self.output_dir)
    
    async def search_emails(self, archive_path: str, query: str, **kwargs) -> Dict[str, Any]:
        """Search emails in archive"""
        try:
            archive = Path(archive_path)
            if not archive.exists():
                archive = config.BASE_DIR / archive.name
                if not archive.exists():
                    raise FileNotFoundError(f"Email archive not found: {archive_path}")
            
            validate_file_size(archive, config.MAX_EMAIL_SIZE)
            emails = await self._parse_email_archive(archive)
            
            query_lower = query.lower()
            matching_emails = []
            
            for email_data in emails:
                if self._email_matches_query(email_data, query_lower):
                    matching_emails.append({
                        "subject": email_data.get("subject", ""),
                        "sender": email_data.get("sender", ""),
                        "date": email_data.get("date", ""),
                        "snippet": email_data.get("content", "")[:200] + "..."
                    })
            
            result = {
                "success": True,
                "message": f"Found {len(matching_emails)} matching emails",
                "total_emails": len(emails),
                "matching_emails": len(matching_emails),
                "results": matching_emails[:50],  # Limit results
                "query": query
            }
            log_to_history("search_emails", "success", str(archive), metadata={"query": query, "matches": len(matching_emails)})
            return result
            
        except Exception as e:
            logger.exception(f"Email search failed")
            log_to_history("search_emails", "error", str(archive_path), error_message=str(e), metadata={"query": query})
            return {
                "success": False,
                "error": str(e),
                "archive_path": archive_path,
                "query": query
            }
    
    async def analyze_communication_patterns(self, archive_path: str, **kwargs) -> Dict[str, Any]:
        """Analyze communication patterns"""
        try:
            archive = Path(archive_path)
            if not archive.exists():
                archive = config.BASE_DIR / archive.name
                if not archive.exists():
                    raise FileNotFoundError(f"Email archive not found: {archive_path}")

            emails = await self._parse_email_archive(archive)
            
            if not emails:
                raise ValueError("No emails found in archive")
            
            sent_count = sum(1 for e in emails if self._is_sent_email(e))
            received_count = len(emails) - sent_count
            
            contacts = {}
            for email_data in emails:
                sender = email_data.get("sender", "")
                if sender and sender not in contacts:
                    contacts[sender] = {"count": 0, "first_contact": None, "last_contact": None}
                
                contacts[sender]["count"] += 1
                date_str = email_data.get("date", "")
                contacts[sender]["last_contact"] = date_str
                if not contacts[sender]["first_contact"]:
                    contacts[sender]["first_contact"] = date_str
            
            top_contacts = sorted(contacts.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
            
            result = {
                "success": True,
                "message": f"Analyzed {len(emails)} emails",
                "total_emails": len(emails),
                "sent_count": sent_count,
                "received_count": received_count,
                "unique_contacts": len(contacts),
                "top_contacts": [{"email": email, **data} for email, data in top_contacts],
                "analysis_date": datetime.now().isoformat()
            }
            log_to_history("analyze_communication_patterns", "success", str(archive), metadata={"total_emails": len(emails)})
            return result
            
        except Exception as e:
            logger.exception(f"Communication analysis failed")
            log_to_history("analyze_communication_patterns", "error", str(archive_path), error_message=str(e))
            return {
                "success": False,
                "error": str(e),
                "archive_path": archive_path
            }
    
    async def _parse_email_archive(self, archive_path: Path) -> List[Dict[str, Any]]:
        """Parse email archive file"""
        emails = []
        try:
            if archive_path.suffix.lower() == '.mbox':
                mbox = mailbox.mbox(str(archive_path))
                for message in mbox:
                    email_data = self._parse_email_message(message)
                    if email_data:
                        emails.append(email_data)
            elif archive_path.suffix.lower() == '.eml':
                with open(archive_path, 'rb') as f:
                    message = BytesParser(policy=default).parse(f)
                    email_data = self._parse_email_message(message)
                    if email_data:
                        emails.append(email_data)
        except Exception as e:
            logger.warning(f"Failed to parse email archive: {e}")
        return emails
    
    def _parse_email_message(self, message) -> Optional[Dict[str, Any]]:
        """Parse individual email message"""
        try:
            return {
                "subject": message.get("Subject", ""),
                "sender": message.get("From", ""),
                "recipients": message.get("To", ""),
                "date": message.get("Date", ""),
                "content": self._get_email_content(message),
                "message_id": message.get("Message-ID", "")
            }
        except Exception:
            return None
    
    def _get_email_content(self, message) -> str:
        """Extract email content"""
        try:
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        return part.get_content()
            else:
                return message.get_content()
        except Exception:
            return ""
    
    def _email_matches_query(self, email_data: Dict[str, Any], query: str) -> bool:
        """Check if email matches search query"""
        searchable_text = " ".join([
            email_data.get("subject", ""),
            email_data.get("sender", ""),
            email_data.get("content", "")
        ]).lower()
        return query in searchable_text
    
    def _is_sent_email(self, email_data: Dict[str, Any]) -> bool:
        """Simple heuristic to determine if email was sent"""
        sender = email_data.get("sender", "").lower()
        return "noreply" not in sender and "no-reply" not in sender

# --- News Processing Service ---
class NewsProcessor:
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR / "news"
        ensure_directory(self.output_dir)
    
    async def add_news_source(self, feed_url: str, category: str = "general", **kwargs) -> Dict[str, Any]:
        """Add RSS news source to the database"""
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                raise ValueError(f"Invalid RSS feed: {feed_url}")
            
            source_data = {
                "url": feed_url,
                "title": feed.feed.get("title", "Unknown"),
                "description": feed.feed.get("description", ""),
                "category": category,
            }
            
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO news_sources (url, title, description, category, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        title=excluded.title,
                        description=excluded.description,
                        category=excluded.category,
                        last_updated=excluded.last_updated
                    """,
                    (source_data["url"], source_data["title"], source_data["description"], source_data["category"], datetime.now().isoformat())
                )
            
            log_to_history("add_news_source", "success", feed_url, metadata=source_data)
            return {
                "success": True,
                "message": f"Added/Updated news source: {source_data['title']}",
                "source_data": source_data
            }
            
        except Exception as e:
            logger.exception(f"Failed to add news source")
            log_to_history("add_news_source", "error", feed_url, error_message=str(e))
            return {
                "success": False,
                "error": str(e),
                "feed_url": feed_url
            }
    
    async def get_trending_topics(self, time_window: str = "24h", **kwargs) -> Dict[str, Any]:
        """Get trending topics from database sources"""
        all_articles = []
        source_count = 0
        
        try:
            with get_db_connection() as conn:
                sources = conn.execute("SELECT id, url, title FROM news_sources WHERE status='active'").fetchall()
                source_count = len(sources)
            
                for source in sources:
                    try:
                        feed = feedparser.parse(source["url"])
                        for entry in feed.entries[:20]:
                            article = {
                                "title": entry.get("title", ""),
                                "summary": entry.get("summary", ""),
                                "published": entry.get("published", ""),
                                "source": source["title"],
                                "link": entry.get("link", ""),
                                "source_id": source["id"],
                                "hash": get_file_hash(entry.get("link", "") + entry.get("title", ""))
                            }
                            all_articles.append(article)
                            
                            conn.execute(
                                """
                                INSERT INTO news_articles (source_id, title, summary, url, published_date, hash)
                                VALUES (?, ?, ?, ?, ?, ?)
                                ON CONFLICT(hash) DO NOTHING
                                """,
                                (article["source_id"], article["title"], article["summary"], article["link"], article["published"], article["hash"])
                            )
                    except Exception as e:
                        logger.warning(f"Failed to fetch from {source['url']}: {e}")
            
            word_counts = {}
            for article in all_articles:
                words = re.findall(r'\b\w+\b', article["title"].lower())
                for word in words:
                    if len(word) > 4 and word not in ['this', 'that', 'with', 'from', 'news']:
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            trending_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            
            with get_db_connection() as conn:
                for word, count in trending_words:
                    conn.execute(
                        """
                        INSERT INTO trending_keywords (keyword, count, category, time_window)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(keyword, time_window, category) DO UPDATE SET
                            count=excluded.count,
                            last_seen=CURRENT_TIMESTAMP
                        """,
                        (word, count, 'general', time_window)
                    )

            result = {
                "success": True,
                "message": f"Found {len(all_articles)} articles from {source_count} sources",
                "total_articles": len(all_articles),
                "trending_keywords": [{"word": word, "mentions": count} for word, count in trending_words],
                "recent_articles": all_articles[:20],
                "time_window": time_window
            }
            
            output_json_path = self.output_dir / "latest_trending.json"
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            log_to_history("get_trending_topics", "success", metadata={"articles_found": len(all_articles), "sources": source_count})
            return result
            
        except Exception as e:
            logger.exception(f"Failed to get trending topics")
            log_to_history("get_trending_topics", "error", error_message=str(e))
            return {
                "success": False,
                "error": str(e),
                "time_window": time_window
            }

# Initialize processors
doc_processor = DocumentProcessor()
img_processor = ImageProcessor()
email_processor = EmailProcessor()
news_processor = NewsProcessor()

# --- MCP Tool Definitions ---
@app.list_tools()
async def list_tools() -> List[types.Tool]:
    """List all available tools"""
    return [
        types.Tool(
            name="convert_document",
            description="Convert document between formats (PDF, DOCX, PPTX, HTML, TXT)",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {"type": "string", "description": "Path to source document"},
                    "target_format": {"type": "string", "enum": config.DOC_FORMATS, "description": "Target format"},
                },
                "required": ["source_path", "target_format"]
            }
        ),
        types.Tool(
            name="process_image",
            description="Process image with resize, enhance, and format conversion",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to source image"},
                    "operations": {"type": "array", "items": {"type": "string", "enum": ["resize", "enhance", "convert"]}, "description": "Operations to apply"},
                    "output_format": {"type": "string", "enum": config.IMG_FORMATS, "default": "png"},
                    "quality": {"type": "integer", "minimum": 1, "maximum": 100, "default": 85},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1}
                },
                "required": ["image_path", "operations"]
            }
        ),
        types.Tool(
            name="search_emails",
            description="Search through email archives (mbox/eml)",
            inputSchema={
                "type": "object",
                "properties": {
                    "archive_path": {"type": "string", "description": "Path to email archive (mbox/eml)"},
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["archive_path", "query"]
            }
        ),
        types.Tool(
            name="analyze_communication_patterns",
            description="Analyze email communication patterns from an archive",
            inputSchema={
                "type": "object",
                "properties": {
                    "archive_path": {"type": "string", "description": "Path to email archive"},
                },
                "required": ["archive_path"]
            }
        ),
        types.Tool(
            name="add_news_source",
            description="Add RSS/news source for monitoring",
            inputSchema={
                "type": "object",
                "properties": {
                    "feed_url": {"type": "string", "description": "RSS feed URL"},
                    "category": {"type": "string", "default": "general"},
                },
                "required": ["feed_url"]
            }
        ),
        types.Tool(
            name="get_trending_topics",
            description="Get trending topics from news sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_window": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                }
            }
        ),
        types.Tool(
            name="list_supported_formats",
            description="Get list of supported formats for each service",
            inputSchema={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "enum": ["document", "image", "email", "all"], "default": "all"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls"""
    try:
        result = None
        
        if name == "convert_document":
            result = await doc_processor.convert_document(**arguments)
        elif name == "process_image":
            result = await img_processor.process_image(**arguments)
        elif name == "search_emails":
            result = await email_processor.search_emails(**arguments)
        elif name == "analyze_communication_patterns":
            result = await email_processor.analyze_communication_patterns(**arguments)
        elif name == "add_news_source":
            result = await news_processor.add_news_source(**arguments)
        elif name == "get_trending_topics":
            result = await news_processor.get_trending_topics(**arguments)
        elif name == "list_supported_formats":
            service = arguments.get("service", "all")
            formats = {}
            if service in ["document", "all"]:
                formats["document"] = config.DOC_FORMATS
            if service in ["image", "all"]:
                formats["image"] = config.IMG_FORMATS
            if service in ["email", "all"]:
                formats["email"] = config.EMAIL_FORMATS
            result = { "success": True, "supported_formats": formats, "service": service }
        else:
            result = { "success": False, "error": f"Unknown tool: {name}" }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
            
    except Exception as e:
        logger.exception(f"Tool call failed")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }, indent=2)
        )]

# --- Resources, Prompts, Status, Main, etc. ---
@app.list_resources()
async def list_resources() -> List[types.Resource]:
    """List available resources"""
    resources = []
    
    # List output files as resources
    for service_dir in ["documents", "images", "email", "news"]:
        service_path = config.OUTPUT_DIR / service_dir
        if service_path.exists():
            for file_path in service_path.iterdir():
                if file_path.is_file():
                    resources.append(types.Resource(
                        uri=f"file://{file_path}",
                        name=f"{service_dir}/{file_path.name}",
                        description=f"Processed {service_dir[:-1]} file",
                        mimeType=_get_mime_type(file_path)
                    ))
    
    return resources

@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content"""
    try:
        if uri.startswith("file://"):
            file_path = Path(uri[7:])  # Remove file:// prefix
            
            # Security check - ensure file is in output directory
            if not str(file_path).startswith(str(config.OUTPUT_DIR)):
                raise ValueError("Access denied: File outside output directory")
            
            if not file_path.exists():
                raise FileNotFoundError(f"Resource not found: {uri}")
            
            # Read file content
            if _is_text_file(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # For binary files, return base64 encoded content
                import base64
                with open(file_path, 'rb') as f:
                    content = base64.b64encode(f.read()).decode('utf-8')
                return f"data:{_get_mime_type(file_path)};base64,{content}"
        
        else:
            raise ValueError(f"Unsupported URI scheme: {uri}")
            
    except Exception as e:
        logger.error(f"Failed to read resource {uri}: {e}")
        raise

# Utility functions for resources
def _get_mime_type(file_path: Path) -> str:
    """Get MIME type for file"""
    try:
        return magic.from_file(str(file_path), mime=True)
    except:
        # Fallback based on extension
        ext = file_path.suffix.lower()
        mime_map = {
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.json': 'application/json'
        }
        return mime_map.get(ext, 'application/octet-stream')

def _is_text_file(file_path: Path) -> bool:
    """Check if file is text-based"""
    text_extensions = {'.txt', '.html', '.htm', '.md', '.json', '.xml', '.csv'}
    return file_path.suffix.lower() in text_extensions

# Health check and status
@app.list_prompts()
async def list_prompts() -> List[types.Prompt]:
    """List available prompts"""
    return [
        types.Prompt(
            name="system_status",
            description="Get system status and statistics",
            arguments=[]
        ),
        types.Prompt(
            name="processing_summary",
            description="Get summary of recent processing activities",
            arguments=[
                types.PromptArgument(
                    name="time_period",
                    description="Time period for summary (1h, 24h, 7d)",
                    required=False
                )
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    """Get prompt content"""
    try:
        if name == "system_status":
            status_info = await _get_system_status()
            return types.GetPromptResult(
                description="Current system status",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"System Status Report:\n\n{json.dumps(status_info, indent=2, default=str)}"
                        )
                    )
                ]
            )
        
        elif name == "processing_summary":
            time_period = arguments.get("time_period", "24h")
            summary_info = await _get_processing_summary(time_period)
            return types.GetPromptResult(
                description=f"Processing summary for {time_period}",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Processing Summary ({time_period}):\n\n{json.dumps(summary_info, indent=2, default=str)}"
                        )
                    )
                ]
            )
        
        else:
            raise ValueError(f"Unknown prompt: {name}")
    
    except Exception as e:
        logger.error(f"Failed to get prompt {name}: {e}")
        return types.GetPromptResult(
            description="Error getting prompt",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Error: {str(e)}"
                    )
                )
            ]
        )

async def _get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    try:
        # Count files in output directories
        file_counts = {}
        total_size = 0
        
        for service_dir in ["documents", "images", "email", "news"]:
            service_path = config.OUTPUT_DIR / service_dir
            if service_path.exists():
                files = list(service_path.iterdir())
                file_counts[service_dir] = len(files)
                total_size += sum(f.stat().st_size for f in files if f.is_file())
            else:
                file_counts[service_dir] = 0
        
        # Check disk space
        disk_usage = shutil.disk_usage(config.BASE_DIR)
        
        # Get DB stats
        db_stats = {}
        with get_db_connection() as conn:
            db_stats["news_sources"] = conn.execute("SELECT COUNT(*) FROM news_sources").fetchone()[0]
            db_stats["news_articles"] = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
            db_stats["trending_keywords"] = conn.execute("SELECT COUNT(*) FROM trending_keywords").fetchone()[0]
            db_stats["processing_history"] = conn.execute("SELECT COUNT(*) FROM processing_history").fetchone()[0]
            
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "directories": {
                "base": str(config.BASE_DIR),
                "temp": str(config.TEMP_DIR),
                "output": str(config.OUTPUT_DIR),
                "cache": str(config.CACHE_DIR)
            },
            "file_counts": file_counts,
            "total_output_size": total_size,
            "disk_usage": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent_used": (disk_usage.used / disk_usage.total) * 100
            },
            "database_stats": db_stats,
            "supported_formats": {
                "documents": config.DOC_FORMATS,
                "images": config.IMG_FORMATS,
                "email": config.EMAIL_FORMATS
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def _get_processing_summary(time_period: str) -> Dict[str, Any]:
    """Get processing summary for time period from DB"""
    try:
        cutoff_hours = {
            "1h": 1,
            "24h": 24,
            "7d": 168
        }.get(time_period, 24)
        
        cutoff_time = datetime.now().timestamp() - (cutoff_hours * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff_time).isoformat()

        summary = {}
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT tool_name, status, COUNT(*) as count
                FROM processing_history
                WHERE created_at >= ?
                GROUP BY tool_name, status
                """,
                (cutoff_iso,)
            ).fetchall()

            for row in rows:
                if row["tool_name"] not in summary:
                    summary[row["tool_name"]] = {"success": 0, "error": 0}
                summary[row["tool_name"]][row["status"]] = row["count"]
        
        total_ops = sum(s["success"] + s["error"] for s in summary.values())
        
        return {
            "time_period": time_period,
            "summary_generated": datetime.now().isoformat(),
            "processing_summary": summary,
            "total_operations": total_ops,
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "time_period": time_period,
            "timestamp": datetime.now().isoformat()
        }

# Cleanup function
async def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        if config.TEMP_DIR.exists():
            for file_path in config.TEMP_DIR.iterdir():
                if file_path.is_file():
                    # Remove files older than 1 hour
                    if file_path.stat().st_mtime < datetime.now().timestamp() - 3600:
                        file_path.unlink()
                        logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Main server function
async def main():
    """Main server function"""
    logger.info("Starting Content & Media Processing MCP Server...")
    
    # Perform initial setup
    try:
        # Create required directories
        config.__post_init__()
        
        # Initialize processors
        global doc_processor, img_processor, email_processor, news_processor
        doc_processor = DocumentProcessor()
        img_processor = ImageProcessor()
        email_processor = EmailProcessor()
        news_processor = NewsProcessor()
        
        # Schedule cleanup task
        asyncio.create_task(_periodic_cleanup())
        
        logger.info("Server initialization complete")
        
        # Run the stdio server
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
            
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

async def _periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await cleanup_temp_files()
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")

# Entry point
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)