#!/usr/bin/env python3
"""
Fixed MCP Protocol Inspector - Follows proper MCP initialization flow
"""

import asyncio
import json
import sys
from pathlib import Path

async def inspect_server_protocol():
    """Try various MCP method calls with proper initialization flow"""
    
    print("Inspecting MCP server protocol with proper initialization...")
    
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "mcp_server.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Step 1: Initialize
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
                "clientInfo": {"name": "inspector", "version": "1.0.0"}
            }
        }
        
        print("Step 1: Sending initialize...")
        await send_message(process, init_message)
        init_response = await receive_message(process)
        
        if init_response.get("error"):
            print(f"Initialize failed: {init_response}")
            return
        
        print("Initialize response received")
        
        # Step 2: Send initialized notification (this was missing!)
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        print("Step 2: Sending initialized notification...")
        await send_message(process, initialized_notification)
        
        # Give server a moment to process
        await asyncio.sleep(0.1)
        
        print("Server should now be ready for requests")
        
        # Step 3: Try tools/list
        tools_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        print("Step 3: Requesting tools/list...")
        await send_message(process, tools_message)
        
        try:
            tools_response = await asyncio.wait_for(receive_message(process), timeout=5.0)
            
            if tools_response.get("error"):
                print(f"Tools list error: {tools_response['error']}")
            else:
                result = tools_response.get("result", {})
                tools = result.get("tools", [])
                print(f"SUCCESS: Found {len(tools)} tools!")
                for tool in tools[:3]:  # Show first 3
                    print(f"  - {tool.get('name', 'unnamed')}: {tool.get('description', '')[:50]}...")
                
        except asyncio.TimeoutError:
            print("Timeout waiting for tools response")
        
        # Step 4: Try other methods if tools/list worked
        if not tools_response.get("error"):
            print("\nTrying other methods...")
            
            # Test resources/list
            resources_msg = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/list",
                "params": {}
            }
            
            await send_message(process, resources_msg)
            try:
                resources_response = await asyncio.wait_for(receive_message(process), timeout=3.0)
                if resources_response.get("error"):
                    print(f"Resources list error: {resources_response['error']['message']}")
                else:
                    resources = resources_response.get("result", {}).get("resources", [])
                    print(f"Resources: Found {len(resources)} resources")
            except asyncio.TimeoutError:
                print("Resources list timeout")
            
            # Test prompts/list
            prompts_msg = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "prompts/list",
                "params": {}
            }
            
            await send_message(process, prompts_msg)
            try:
                prompts_response = await asyncio.wait_for(receive_message(process), timeout=3.0)
                if prompts_response.get("error"):
                    print(f"Prompts list error: {prompts_response['error']['message']}")
                else:
                    prompts = prompts_response.get("result", {}).get("prompts", [])
                    print(f"Prompts: Found {len(prompts)} prompts")
            except asyncio.TimeoutError:
                print("Prompts list timeout")
        
        # Check stderr for any additional info
        try:
            stderr_data = await asyncio.wait_for(process.stderr.read(1024), timeout=1.0)
            if stderr_data:
                stderr_text = stderr_data.decode()
                if "ERROR" in stderr_text or "WARNING" in stderr_text:
                    print(f"\nServer stderr:")
                    print(stderr_text)
        except asyncio.TimeoutError:
            pass
        
        process.terminate()
        await process.wait()
        
        print("\nProtocol inspection complete!")
        
    except Exception as e:
        print(f"Inspection failed: {e}")

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

if __name__ == "__main__":
    asyncio.run(inspect_server_protocol())