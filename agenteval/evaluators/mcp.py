"""Native MCP client validation over official stdio and legacy SSE transports.

Install the optional client SDK with ``pip install agenteval-bench[mcp]``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field


class MCPCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPValidationResult(BaseModel):
    passed: bool
    transport: str
    discrepancies: list[str] = Field(default_factory=list)
    available_tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    call_results: list[dict[str, Any]] = Field(default_factory=list)


class MCPEvaluator:
    """Initialize an MCP client session, validate published tool schemas, and call tools."""

    @staticmethod
    def compare_schema(name: str, expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        annotations = {
            "title",
            "description",
            "examples",
            "default",
            "deprecated",
            "readOnly",
            "writeOnly",
        }

        def compare(expected_node: Any, actual_node: Any, path: str, root: bool = False) -> None:
            if not isinstance(expected_node, dict) or not isinstance(actual_node, dict):
                if expected_node != actual_node:
                    issues.append(f"{name}: schema mismatch at '{path}'")
                return
            if "type" in expected_node and expected_node.get("type") != actual_node.get("type"):
                issues.append(
                    f"{name}: type mismatch for '{path or '$'}': expected {expected_node.get('type')}, got {actual_node.get('type')}"
                )
            if "required" in expected_node or "required" in actual_node:
                expected_required = set(expected_node.get("required", []))
                actual_required = set(actual_node.get("required", []))
                if expected_required != actual_required:
                    label = "required fields differ" if root else "nested required fields differ"
                    issues.append(
                        f"{name}: {label} at '{path}': expected {sorted(expected_required)}, got {sorted(actual_required)}"
                    )
            expected_properties = expected_node.get("properties", {})
            actual_properties = actual_node.get("properties", {})
            for prop, spec in expected_properties.items():
                prop_path = f"{path}.{prop}" if path else prop
                if prop not in actual_properties:
                    issues.append(f"{name}: missing property '{prop_path}'")
                else:
                    compare(spec, actual_properties[prop], prop_path)
            for keyword in set(expected_node) | set(actual_node):
                if keyword in annotations or keyword in {"type", "required", "properties"}:
                    continue
                missing = object()
                left = expected_node.get(keyword, missing)
                right = actual_node.get(keyword, missing)
                keyword_path = f"{path}.{keyword}" if path else keyword
                if left is missing or right is missing:
                    issues.append(f"{name}: constraint presence differs for '{keyword_path}'")
                    continue
                if keyword == "items" and isinstance(left, dict) and isinstance(right, dict):
                    compare(left, right, keyword_path)
                elif left != right:
                    issues.append(
                        f"{name}: constraint mismatch for '{keyword_path}': expected {left!r}, got {right!r}"
                    )

        compare(expected, actual, "", root=True)
        return issues

    async def validate_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        expected_schemas: dict[str, dict[str, Any]] | None = None,
        calls: list[MCPCall] | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float = 30,
    ) -> MCPValidationResult:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ImportError(
                "MCP support requires the optional extra: pip install agenteval-bench[mcp]"
            ) from exc
        parameters = StdioServerParameters(command=command, args=args or [], env=env)
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            return await self._evaluate_session(
                session, "stdio", expected_schemas or {}, calls or [], timeout_seconds
            )

    async def validate_sse(
        self,
        url: str,
        expected_schemas: dict[str, dict[str, Any]] | None = None,
        calls: list[MCPCall] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30,
    ) -> MCPValidationResult:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise ImportError(
                "MCP support requires the optional extra: pip install agenteval-bench[mcp]"
            ) from exc
        async with (
            sse_client(url, headers=headers, timeout=timeout_seconds) as (
                read_stream,
                write_stream,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            return await self._evaluate_session(
                session, "sse", expected_schemas or {}, calls or [], timeout_seconds
            )

    async def _evaluate_session(
        self,
        session: Any,
        transport: str,
        expected_schemas: dict[str, dict[str, Any]],
        calls: list[MCPCall],
        timeout_seconds: float,
    ) -> MCPValidationResult:
        discrepancies: list[str] = []
        call_results: list[dict[str, Any]] = []
        await asyncio.wait_for(session.initialize(), timeout=timeout_seconds)
        listed = await asyncio.wait_for(session.list_tools(), timeout=timeout_seconds)
        available: dict[str, dict[str, Any]] = {}
        for tool in listed.tools:
            if hasattr(tool, "model_dump"):
                dumped = tool.model_dump(by_alias=True, exclude_none=True)
            else:
                dumped = {}
            schema = dumped.get("inputSchema", dumped.get("input_schema"))
            if schema is None:
                schema = getattr(tool, "inputSchema", getattr(tool, "input_schema", {}))
            available[tool.name] = schema if isinstance(schema, dict) else {}
        for name, expected in expected_schemas.items():
            if name not in available:
                discrepancies.append(f"MCP server does not expose expected tool '{name}'")
            else:
                discrepancies.extend(self.compare_schema(name, expected, available[name]))
        for request in calls:
            if request.name not in available:
                discrepancies.append(f"cannot call missing MCP tool '{request.name}'")
                continue
            result = await asyncio.wait_for(
                session.call_tool(request.name, arguments=request.arguments),
                timeout=timeout_seconds,
            )
            serialized = (
                result.model_dump(by_alias=True, exclude_none=True)
                if hasattr(result, "model_dump")
                else {"result": str(result)}
            )
            call_results.append({"name": request.name, "result": serialized})
            if getattr(result, "isError", getattr(result, "is_error", False)):
                discrepancies.append(f"MCP tool '{request.name}' returned an error")
        return MCPValidationResult(
            passed=not discrepancies,
            transport=transport,
            discrepancies=discrepancies,
            available_tools=available,
            call_results=call_results,
        )
