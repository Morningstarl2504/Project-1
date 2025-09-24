#!/usr/bin/env python3
"""
Complete Test Implementation for MCP Content & Media Processing Server
"""

import asyncio
import json
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import tempfile
import shutil
import uuid
import inspect

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the client classes (assuming they're in the same directory)
try:
    from mcp_client import MCPContentClient, ContentProcessor
except ImportError as e:
    logger.error(f"❌ Error importing mcp_client: {e}")
    logger.error("💡 Make sure mcp_client.py is in the same directory as this script.")
    sys.exit(1)


class TestSuite:
    """Complete test suite for MCP server functionality"""
    
    def __init__(self, server_path: str):
        self.server_path = Path(server_path)
        self.temp_dir = None
        self.test_results = {}
        self.test_files = {}
        self.processor = None
        
        if not self.server_path.exists():
            raise FileNotFoundError(f"Server script not found: {server_path}")
        
    async def run_dependency_check(self):
        """Check if all required dependencies are installed"""
        print("📋 Checking Dependencies")
        print("=" * 40)
        
        required_modules = {
            "mcp": "MCP Protocol",
            "PIL": "Pillow (Image processing)",
            "pypandoc": "Pandoc wrapper",
            "docx": "python-docx",
            "PyPDF2": "PDF processing",
            "bs4": "BeautifulSoup4",
            "markdown": "Markdown processor",
            "feedparser": "RSS feed parser",
            "requests": "HTTP requests",
            "magic": "python-magic",
        }
        
        all_ok = True
        for module, description in required_modules.items():
            try:
                __import__(module)
                print(f"   ✅ {description}")
            except ImportError:
                print(f"   ❌ {description} - MISSING")
                all_ok = False
        
        if all_ok:
            print("\n✅ All dependencies are installed!")
        else:
            print("\n❌ Some dependencies are missing. Please install them using pip.")
        return all_ok

    async def run_full_test_suite(self):
        """Run the complete test suite"""
        print("\n🧪 Full Test Suite")
        print("=" * 60)
        
        await self.setup_test_environment()
        
        try:
            # Connect once for all tests
            async with MCPContentClient(str(self.server_path)) as client:
                self.processor = ContentProcessor(client)
                
                # Run all tests dynamically
                test_methods = [method for name, method in inspect.getmembers(self, inspect.ismethod) if name.startswith("test_")]
                
                for test_method in test_methods:
                    test_name = test_method.__name__.replace("test_", "")
                    try:
                        result = await test_method()
                        self.test_results[test_name] = result
                    except Exception as e:
                        self.test_results[test_name] = {"status": "FAIL", "message": str(e)}
        except Exception as e:
            logger.error(f"Test suite failed to run: {e}")
            self.test_results = {"suite_error": {"status": "FAIL", "message": f"Suite failed to run: {e}"}}
        finally:
            await self.cleanup_test_environment()
            self.print_summary()

    async def run_quick_test(self):
        """Run a quick connection test"""
        print("\n🚀 Quick Test")
        print("=" * 40)
        try:
            async with MCPContentClient(str(self.server_path)) as client:
                tools = await client.list_tools()
                if tools:
                    print(f"✅ Quick test passed! Found {len(tools)} tools.")
                    return True
                else:
                    print("❌ Quick test failed. No tools found.")
                    return False
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False

    async def setup_test_environment(self):
        """Setup test environment"""
        print("\n🔧 Setting up test environment...")
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="mcp_test_"))
            self.test_files = self.create_test_files()
            print(f"   ✅ Test environment created: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to set up test environment: {e}")
            raise

    def create_test_files(self) -> Dict[str, Path]:
        """Create test files for processing"""
        files = {}
        
        test_doc = self.temp_dir / "test_document.md"
        with open(test_doc, 'w', encoding='utf-8') as f:
            f.write("# Test Document\n\nThis is a test document for MCP processing.\n")
            
        test_img = self.temp_dir / "test_image.png"
        try:
            from PIL import Image
            img = Image.new('RGB', (200, 200), color='blue')
            img.save(test_img)
        except ImportError:
            pass # Skip image creation if PIL is not installed
            
        test_email = self.temp_dir / "test.eml"
        with open(test_email, 'w', encoding='utf-8') as f:
            f.write("From: test@example.com\nSubject: Test Email\n\nThis is a test email message.\n")

        files["doc"] = test_doc
        files["img"] = test_img
        files["email"] = test_email
        return files

    async def test_server_connection(self):
        """Test basic server connection"""
        print("\n🔗 Testing server connection...")
        try:
            async with MCPContentClient(str(self.server_path)) as client:
                tools = await client.list_tools()
                if tools:
                    print(f"   ✅ Connection successful ({len(tools)} tools available)")
                    return {"status": "PASS", "message": f"Connected successfully, found {len(tools)} tools"}
                else:
                    print("   ❌ Connection failed - no tools available")
                    return {"status": "FAIL", "message": "No tools available"}
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            return {"status": "FAIL", "message": str(e)}

    async def test_tools_listing(self):
        """Test tools listing functionality"""
        print("\n🛠️  Testing tools listing...")
        try:
            async with MCPContentClient(str(self.server_path)) as client:
                tools = await client.list_tools()
                expected_tools = ["convert_document", "process_image", "search_emails", "add_news_source", "get_trending_topics"]
                found_tool_names = {tool.get("name") for tool in tools}
                
                missing_tools = [tool for tool in expected_tools if tool not in found_tool_names]
                
                if not missing_tools:
                    print(f"   ✅ All expected tools found ({len(tools)} total)")
                    return {"status": "PASS", "message": f"All {len(expected_tools)} expected tools found"}
                else:
                    print(f"   ⚠️  Missing tools: {', '.join(missing_tools)}")
                    return {"status": "PARTIAL", "message": f"Missing tools: {missing_tools}"}
        except Exception as e:
            print(f"   ❌ Tools listing failed: {e}")
            return {"status": "FAIL", "message": str(e)}

    async def test_system_status(self):
        """Test system status functionality"""
        print("\n📊 Testing system status...")
        try:
            async with MCPContentClient(str(self.server_path)) as client:
                status = await client.get_system_status()
                if "status" in status and status["status"] == "ok":
                    print(f"   ✅ System status retrieved: {status.get('status', 'unknown')}")
                    return {"status": "PASS", "message": f"System status retrieved: {status['status']}"}
                else:
                    print(f"   ❌ Invalid status response: {status}")
                    return {"status": "FAIL", "message": f"Invalid status response: {status}"}
        except Exception as e:
            print(f"   ❌ System status failed: {e}")
            return {"status": "FAIL", "message": str(e)}

    async def test_document_processing(self):
        """Test document processing functionality"""
        print("\n📄 Testing document processing...")
        if not self.test_files.get("doc", Path(".")):
            return {"status": "SKIP", "message": "Test document not created"}
            
        try:
            result = await self.processor.convert_document(str(self.test_files["doc"]), "html")
            if result.get("success"):
                print("   ✅ Document conversion successful")
                return {"status": "PASS", "message": f"Document converted successfully to {result.get('target_format')}"}
            else:
                print(f"   ❌ Document conversion failed: {result.get('error', 'Unknown error')}")
                return {"status": "FAIL", "message": result.get('error', 'Unknown error')}
        except Exception as e:
            print(f"   ❌ Document processing test failed: {e}")
            return {"status": "FAIL", "message": str(e)}

    async def test_image_processing(self):
        """Test image processing functionality"""
        print("\n🖼️  Testing image processing...")
        if not self.test_files.get("img", Path(".")):
            return {"status": "SKIP", "message": "Test image not created"}
            
        try:
            result = await self.processor.process_image(str(self.test_files["img"]), ["resize", "enhance"], width=100, height=100)
            if result.get("success"):
                print("   ✅ Image processing successful")
                return {"status": "PASS", "message": f"Image processed with {len(result.get('operations_applied', []))} operations"}
            else:
                print(f"   ❌ Image processing failed: {result.get('error', 'Unknown error')}")
                return {"status": "FAIL", "message": result.get('error', 'Unknown error')}
        except Exception as e:
            print(f"   ❌ Image processing test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
            
    async def test_email_processing(self):
        """Test email processing functionality"""
        print("\n📧 Testing email processing...")
        if not self.test_files.get("email", Path(".")):
            return {"status": "SKIP", "message": "Test email not created"}
            
        try:
            result = await self.processor.search_emails(str(self.test_files["email"]), "test")
            if result.get("success"):
                print("   ✅ Email search successful")
                return {"status": "PASS", "message": f"Email search successful, found {result.get('total_results', 0)} results"}
            else:
                print(f"   ❌ Email search failed: {result.get('error', 'Unknown error')}")
                return {"status": "FAIL", "message": result.get('error', 'Unknown error')}
        except Exception as e:
            print(f"   ❌ Email processing test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
            
    async def test_news_functionality(self):
        """Test news functionality"""
        print("\n📰 Testing news functionality...")
        try:
            result = await self.processor.add_news_source("https://rss.cnn.com/rss/edition.rss", "test")
            if result.get("success"):
                print("   ✅ News source added successfully")
                return {"status": "PASS", "message": "News source added successfully"}
            else:
                print(f"   ❌ News source addition failed: {result.get('error', 'Unknown error')}")
                return {"status": "FAIL", "message": result.get('error', 'Unknown error')}
        except Exception as e:
            print(f"   ❌ News functionality test failed: {e}")
            return {"status": "FAIL", "message": str(e)}

    async def test_resource_management(self):
        """Test resource management functionality"""
        print("\n📁 Testing resource management...")
        try:
            async with MCPContentClient(str(self.server_path)) as client:
                resources = await client.list_resources()
                print(f"   ✅ Resource listing successful ({len(resources)} resources)")
                return {"status": "PASS", "message": f"Resource listing successful, {len(resources)} resources found"}
        except Exception as e:
            print(f"   ❌ Resource management test failed: {e}")
            return {"status": "FAIL", "message": str(e)}

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "FAIL")
        partial_tests = sum(1 for result in self.test_results.values() if result["status"] == "PARTIAL")
        skipped_tests = sum(1 for result in self.test_results.values() if result["status"] == "SKIP")
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️ Partial: {partial_tests}")
        print(f"⭕ Skipped: {skipped_tests}")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "⚠️", "SKIP": "⭕"}.get(result["status"], "❓")
            print(f"  {status_emoji} {test_name}: {result['message']}")
        
        if failed_tests == 0 and partial_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! Server is working correctly.")
        elif failed_tests == 0:
            print(f"\n⚠️ TESTS MOSTLY PASSED with {partial_tests} partial results.")
        else:
            print(f"\n❌ {failed_tests} TESTS FAILED. Check server configuration.")
        
        print("=" * 60)
    
    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        print("\n🧹 Cleaning up test environment...")
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"   ✅ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"   ⚠️ Cleanup failed: {e}")

async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test MCP Content & Media Processing Server")
    parser.add_argument("--server", required=True, help="Path to server script")
    parser.add_argument("--deps", action="store_true", help="Check dependencies only")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    
    args = parser.parse_args()
    
    suite = TestSuite(args.server)
    
    if args.deps:
        await suite.run_dependency_check()
    elif args.quick:
        deps_ok = await suite.run_dependency_check()
        if deps_ok:
            await suite.run_quick_test()
    elif args.full:
        deps_ok = await suite.run_dependency_check()
        if deps_ok:
            await suite.run_full_test_suite()
    else:
        # Default: run dependency check then full suite
        deps_ok = await suite.run_dependency_check()
        if deps_ok:
            await suite.run_full_test_suite()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n\n⏹️ Test run cancelled by user.")
        sys.exit(0)