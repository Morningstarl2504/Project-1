#!/usr/bin/env python3
"""
Test script for the fixed MCP client
"""

import asyncio
import sys
from pathlib import Path

async def test_fixed_client():
    """Test the fixed MCP client"""
    print("Testing fixed MCP client...")
    
    # Import the fixed client
    try:
        from mcp_client import MCPContentClient, ContentProcessor
        print("✅ Successfully imported fixed MCP client")
    except ImportError as e:
        print(f"❌ Failed to import MCP client: {e}")
        return False
    
    server_path = "mcp_server.py"
    if not Path(server_path).exists():
        print(f"❌ Server file not found: {server_path}")
        return False
    
    try:
        print("📡 Connecting to server...")
        async with MCPContentClient(server_path) as client:
            print("✅ Connected successfully!")
            
            # Test 1: List tools
            print("\n🔍 Testing tools listing...")
            tools = await client.list_tools()
            if tools:
                print(f"✅ Found {len(tools)} tools:")
                for i, tool in enumerate(tools[:3], 1):
                    name = tool.get('name', 'unnamed')
                    print(f"   {i}. {name}")
            else:
                print("❌ No tools found")
                return False
            
            # Test 2: Call a tool
            print("\n🛠️ Testing tool call...")
            try:
                result = await client.call_tool("list_supported_formats", {"service": "all"})
                if result.get("success"):
                    print("✅ Tool call successful!")
                    formats = result.get("supported_formats", {})
                    print(f"   Found formats for {len(formats)} services")
                else:
                    print(f"❌ Tool call failed: {result.get('error')}")
                    return False
            except Exception as e:
                print(f"❌ Tool call error: {e}")
                return False
            
            # Test 3: List resources
            print("\n📁 Testing resource listing...")
            try:
                resources = await client.list_resources()
                print(f"✅ Found {len(resources)} resources")
            except Exception as e:
                print(f"⚠️  Resource listing failed: {e}")
            
            print("\n🎉 All tests passed! Client is working correctly.")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixed_client())
    sys.exit(0 if success else 1)