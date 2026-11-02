import asyncio
import sys

import pytest

from agenteval.evaluators.mcp import MCPCall, MCPEvaluator


def test_official_mcp_sdk_stdio_validation_and_tool_call():
    pytest.importorskip("mcp")
    server = """from mcp.server.fastmcp import FastMCP
mcp = FastMCP("eval-test")
@mcp.tool()
def greet(name: str) -> str:
    return "hello " + name
mcp.run(transport="stdio")
"""
    result = asyncio.run(
        MCPEvaluator().validate_stdio(
            sys.executable,
            ["-c", server],
            expected_schemas={
                "greet": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            calls=[MCPCall(name="greet", arguments={"name": "Hermes"})],
            timeout_seconds=20,
        )
    )
    assert result.passed, result.discrepancies
    assert result.transport == "stdio"
    assert result.call_results
