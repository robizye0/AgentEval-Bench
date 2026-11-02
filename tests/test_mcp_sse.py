import asyncio
import socket
import subprocess
import sys
import time

import pytest

from agenteval.evaluators.mcp import MCPEvaluator


def test_official_mcp_sdk_sse_schema_validation():
    pytest.importorskip("mcp")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = f"""from mcp.server.fastmcp import FastMCP
mcp = FastMCP("sse-test", host="127.0.0.1", port={port})
@mcp.tool()
def ping(value: str) -> str:
    return value
mcp.run(transport="sse")
"""
    process = subprocess.Popen(
        [sys.executable, "-c", server], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        ready = False
        for _ in range(100):
            if process.poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    ready = True
                    break
            except OSError:
                time.sleep(0.05)
        assert ready, "MCP SSE test server did not start"
        result = asyncio.run(
            MCPEvaluator().validate_sse(
                f"http://127.0.0.1:{port}/sse",
                expected_schemas={
                    "ping": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    }
                },
                timeout_seconds=10,
            )
        )
        assert result.passed, result.discrepancies
        assert result.transport == "sse"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
