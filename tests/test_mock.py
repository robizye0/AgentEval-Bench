import asyncio

import pytest

from agenteval.evaluators.resilience import InjectedFailure
from agenteval.mock.tool_registry import ToolRegistry


def test_registry_validates_schema_and_runs_async_tool():
    class Args(__import__("pydantic").BaseModel):
        query: str

    async def handler(query: str):
        return {"query": query}

    registry = ToolRegistry()
    registry.register("search", handler, Args)
    assert asyncio.run(registry.call("search", {"query": "x"})) == {"query": "x"}
    with pytest.raises(ValueError):
        asyncio.run(registry.call("search", {"query": "x", "extra": 1}))


def test_registry_injects_simulated_failure():
    registry = ToolRegistry()
    registry.register("search", lambda: "ok")
    with pytest.raises(InjectedFailure):
        asyncio.run(registry.call("search", {}, failure="http_500"))
