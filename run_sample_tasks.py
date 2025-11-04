import asyncio
import json
import sys
from pathlib import Path

# Configuration - adjust filenames chosen for the tasks
BASE = Path(__file__).parent
SERVER_SCRIPT = BASE / 'mcp_server.py'
OUT_DIR = BASE / 'client_output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Files to use (from repo)
DOC1 = BASE / 'Utkarsh_Gupta_Internship_report.docx'
PPT1 = BASE / 'Utkarsh_Gupta_Industrial_Internship_Presentation.pptx'
IMG1 = BASE / 'output' / 'images' / 'wallpaperflare.com_wallpaper_processed.png'

# Email sample - we'll create a minimal .eml file
EMAIL_SAMPLE = BASE / 'client_temp' / 'sample_email.eml'
EMAIL_SAMPLE.parent.mkdir(parents=True, exist_ok=True)

RSS_EXAMPLE = 'http://feeds.bbci.co.uk/news/rss.xml'

async def run():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(SERVER_SCRIPT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def send(obj):
        s = json.dumps(obj) + "\n"
        proc.stdin.write(s.encode())
        await proc.stdin.drain()

    async def recv(timeout=10.0):
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            stderr = await proc.stderr.read()
            raise RuntimeError(f"Timeout waiting for response. STDERR: {stderr.decode(errors='replace')}")
        if not line:
            stderr = await proc.stderr.read()
            raise RuntimeError(f"No response from server. STDERR: {stderr.decode(errors='replace')}")
        return json.loads(line.decode().strip())

    try:
        # Initialize
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sample-runner", "version": "0.1"}
            }
        }
        await send(init_msg)
        init_resp = await recv(10.0)
        print('Initialized server: ', init_resp.get('result', {}).get('serverInfo', {}))

        # helper to call tools
        async def call_tool(name, arguments, out_name):
            call_msg = {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments}
            }
            await send(call_msg)
            resp = await recv(30.0)
            # save
            fp = OUT_DIR / out_name
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(resp, f, indent=2, ensure_ascii=False)
            print(f"Saved response for {name} -> {fp}")
            return resp

        # 1) Document conversions
        if DOC1.exists():
            await call_tool('convert_document', {"source_path": str(DOC1), "target_format": "pdf"}, 'convert_docx_to_pdf.json')
        else:
            print(f"Document not found: {DOC1}")

        if PPT1.exists():
            await call_tool('convert_document', {"source_path": str(PPT1), "target_format": "pdf"}, 'convert_pptx_to_pdf.json')
        else:
            print(f"Presentation not found: {PPT1}")

        # 2) Image processing: resize (width=800), enhance, convert to jpeg
        if IMG1.exists():
            await call_tool('process_image', {"image_path": str(IMG1), "operations": ["resize", "enhance", "convert"], "output_format": "jpeg", "width": 800}, 'process_image_wallpaper.json')
        else:
            print(f"Image not found: {IMG1}")

        # 3) Email: create a simple .eml and run search + analyze
        sample_eml_text = (
            "From: Alice <alice@example.com>\n"
            "To: Bob <bob@example.com>\n"
            "Subject: Test email for MCP\n"
            "Date: Fri, 01 Nov 2025 10:00:00 +0000\n"
            "Message-ID: <sample1@example.com>\n"
            "\n"
            "Hello Bob,\n\nThis is a test email to demonstrate email parsing and searching.\nRegards,\nAlice\n"
        )
        with open(EMAIL_SAMPLE, 'w', encoding='utf-8') as f:
            f.write(sample_eml_text)
        print(f"Created sample email archive at {EMAIL_SAMPLE}")

        await call_tool('search_emails', {"archive_path": str(EMAIL_SAMPLE), "query": "test"}, 'search_emails_sample.json')
        await call_tool('analyze_communication_patterns', {"archive_path": str(EMAIL_SAMPLE)}, 'analyze_emails_sample.json')

        # 4) News: add RSS source and get trending topics
        await call_tool('add_news_source', {"feed_url": RSS_EXAMPLE, "category": "world"}, 'add_news_source_bbc.json')
        await call_tool('get_trending_topics', {"time_window": "24h"}, 'get_trending_topics_24h.json')

        print('\nAll tasks completed. Outputs saved to ' + str(OUT_DIR))

    except Exception as e:
        print('ERROR DURING RUN:', e)
    finally:
        # Terminate server
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except Exception:
            proc.kill()
            await proc.wait()

if __name__ == '__main__':
    asyncio.run(run())
