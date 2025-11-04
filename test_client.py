import asyncio
import json
import sys
from pathlib import Path

async def main():
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).parent / 'mcp_server.py'),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        async def send(obj):
            s = json.dumps(obj) + "\n"
            proc.stdin.write(s.encode())
            await proc.stdin.drain()

        async def recv(timeout=5.0):
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                stderr = await proc.stderr.read()
                raise RuntimeError(f"Timeout waiting for response. STDERR: {stderr.decode(errors='replace')}")

            if not line:
                stderr = await proc.stderr.read()
                raise RuntimeError(f"No response from server. STDERR: {stderr.decode(errors='replace')}")

            return json.loads(line.decode().strip())

        # Send initialize
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1"}
            }
        }

        await send(init_msg)
        init_resp = await recv(10.0)
        print("INIT RESPONSE:\n", json.dumps(init_resp, indent=2))

        # Request tools list
        tools_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        await send(tools_msg)
        tools_resp = await recv(10.0)
        print("TOOLS LIST RESPONSE:\n", json.dumps(tools_resp, indent=2))

        # Shut down
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
