#!/usr/bin/env python3
"""
Fixed diagnostic test to check if the MCP server works correctly
"""

import asyncio
import subprocess
import sys
import json
import os
from pathlib import Path

try:
    from PIL import Image
    print("✅ Pillow is installed and can be imported successfully!")
except ImportError as e:
    print(f"❌ Error: Pillow is not installed or cannot be imported: {e}")
    sys.exit(1)

async def test_server_directly():
    """Test the server by running it directly and sending basic MCP messages"""

    print("🔍 Testing MCP server directly...")

    # Check if server file exists
    server_path = Path("mcp_server.py")
    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        return False

    try:
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(server_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        print("✅ Server process started")
        
        # Send initialize message
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        # Send the message
        await send_message(process, init_message)
        print("📤 Sent initialize message")

        # Receive the response
        response = await receive_message(process)
        print(f"📥 Received response: {json.dumps(response, indent=2)}")
        
        if "error" in response:
            print("❌ Initialization failed. Check the server output for details.")
            return False
            
        print("✅ Server initialized successfully!")
        
        # Send the 'initialized' notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        await send_message(process, initialized_notification)
        print("📤 Sent initialized notification")

        # Wait a moment for server to process
        await asyncio.sleep(0.2)

        # Now send the tools list request - FIXED FORMAT
        tools_list_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
            # Note: removed empty params - some MCP servers don't like empty params
        }

        # Send the message
        await send_message(process, tools_list_message)
        print("📤 Sent tools/list message")

        # Receive the tools list response with timeout
        try:
            response = await asyncio.wait_for(receive_message(process), timeout=5.0)
            print(f"📥 Tools response: {json.dumps(response, indent=2)}")
            
            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                print(f"✅ Tools list successful! Found {len(tools)} tools:")
                for i, tool in enumerate(tools[:5]):  # Show first 5 tools
                    print(f"   {i+1}. {tool.get('name', 'unnamed')}")
                return True
            elif "error" in response:
                print(f"❌ Tools list failed with error: {response['error']}")
                return False
            else:
                print(f"❌ Unexpected response format: {response}")
                return False
        except asyncio.TimeoutError:
            print("❌ Timeout waiting for tools list response")
            return False
        
    except Exception as e:
        print(f"❌ Server communication failed: {e}")
        return False
    finally:
        if 'process' in locals() and process.returncode is None:
            process.terminate()
            await process.wait()

async def send_message(process, message):
    """Send JSON message to process"""
    json_str = json.dumps(message) + '\n'
    process.stdin.write(json_str.encode())
    await process.stdin.drain()

async def receive_message(process):
    """Receive JSON message from process"""
    line = await process.stdout.readline()
    if not line:
        raise Exception("No response from server")
    return json.loads(line.decode().strip())

async def check_server_imports():
    """Check if the server can import all required modules"""
    print("\n🔍 Checking server dependencies...")
    
    try:
        # Try to import the server and see if it has import errors
        result = subprocess.run([
            sys.executable, "-c", 
            "import mcp_server; print('Server imports OK')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Server imports are OK")
            return True
        else:
            print(f"❌ Server import failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Server import check timed out")
        return False
    except Exception as e:
        print(f"❌ Import check failed: {e}")
        return False

async def check_basic_dependencies():
    """Check if basic dependencies are installed"""
    print("\n🔍 Checking basic dependencies...")
    
    required_modules = [
        "mcp",
        "PIL", 
        "pypandoc",
        "docx",
        "PyPDF2",
        "bs4",
        "markdown",
        "feedparser",
        "requests",
        "magic"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    else:
        print("\n✅ All dependencies are installed")
        return True

async def create_test_file():
    """Create a simple test file for conversion"""
    test_file = Path("test.md")
    content = """# Test Document

This is a **test** document for MCP processing.

## Features
- Markdown formatting
- Multiple sections
- Lists and emphasis

Content for testing document conversion functionality.
"""
    
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created test file: {test_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create test file: {e}")
        return False

async def test_direct_tool_call():
    """Test calling a tool directly"""
    print("\n🔧 Testing direct tool call...")
    
    try:
        # Start server
        process = await asyncio.create_subprocess_exec(
            sys.executable, "mcp_server.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Initialize
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        await send_message(process, init_msg)
        init_response = await receive_message(process)
        
        if "error" in init_response:
            print(f"❌ Initialization failed: {init_response['error']}")
            return False
        
        # Send initialized notification
        await send_message(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        await asyncio.sleep(0.2)
        
        # Try to call list_supported_formats tool
        tool_call = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "list_supported_formats",
                "arguments": {"service": "all"}
            }
        }
        
        await send_message(process, tool_call)
        
        try:
            response = await asyncio.wait_for(receive_message(process), timeout=5.0)
            if "result" in response:
                print("✅ Tool call successful!")
                print(f"📥 Response: {json.dumps(response['result'], indent=2)}")
                return True
            else:
                print(f"❌ Tool call failed: {response}")
                return False
        except asyncio.TimeoutError:
            print("❌ Tool call timed out")
            return False
    
    except Exception as e:
        print(f"❌ Tool call test failed: {e}")
        return False
    finally:
        if 'process' in locals() and process.returncode is None:
            process.terminate()
            await process.wait()

async def main():
    """Run all diagnostic tests"""
    print("=" * 60)
    print("MCP Server Diagnostic Tests")
    print("=" * 60)
    
    # Test 1: Check dependencies
    deps_ok = await check_basic_dependencies()
    
    # Test 2: Check server imports
    imports_ok = await check_server_imports() if deps_ok else False
    
    # Test 3: Create test file
    test_file_ok = await create_test_file()
    
    # Test 4: Test server directly
    server_ok = await test_server_directly() if imports_ok else False
    
    # Test 5: Test direct tool call
    tool_call_ok = await test_direct_tool_call() if server_ok else False
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Dependencies: {'✅ OK' if deps_ok else '❌ FAILED'}")
    print(f"Server imports: {'✅ OK' if imports_ok else '❌ FAILED'}")
    print(f"Test file creation: {'✅ OK' if test_file_ok else '❌ FAILED'}")
    print(f"Server communication: {'✅ OK' if server_ok else '❌ FAILED'}")
    print(f"Tool calling: {'✅ OK' if tool_call_ok else '❌ FAILED'}")
    
    if not deps_ok:
        print("\n🔧 Next steps: Install missing dependencies")
    elif not imports_ok:
        print("\n🔧 Next steps: Fix server import errors")
    elif not server_ok:
        print("\n🔧 Next steps: Debug server communication protocol")
    elif not tool_call_ok:
        print("\n🔧 Next steps: Debug tool calling mechanism")
    else:
        print("\n🎉 All tests passed! Server should work.")

if __name__ == "__main__":
    asyncio.run(main())