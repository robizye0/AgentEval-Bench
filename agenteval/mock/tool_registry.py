"""Async registry for mock tools and MCP-compatible callables."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable[..., Any], type[BaseModel] | None]] = {}

    def register(
        self, name: str, handler: Callable[..., Any], schema: type[BaseModel] | None = None
    ) -> None:
        if not name or name in self._tools:
            raise ValueError(f"tool name is empty or already registered: {name}")
        self._tools[name] = (handler, schema)

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        delay: float = 0,
        failure: str | None = None,
        timeout: float | None = None,
        jitter_ms: int = 0,
        spike_probability: float = 0,
        latency_spike_ms: float = 3000,
        random_seed: int = 0,
    ) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")

        async def execute() -> Any:
            if delay or jitter_ms or spike_probability:
                from agenteval.evaluators.resilience import ResilienceEvaluator

                await ResilienceEvaluator.inject_latency(
                    base_latency_ms=delay * 1000,
                    jitter_ms=jitter_ms,
                    spike_probability=spike_probability,
                    spike_latency_ms=latency_spike_ms,
                    seed=random_seed,
                )
            if failure:
                from agenteval.evaluators.resilience import ResilienceEvaluator

                ResilienceEvaluator.inject(failure)
            handler, schema = self._tools[name]
            if schema:
                unknown = set(arguments) - set(schema.model_fields)
                if unknown:
                    raise ValueError(f"unexpected tool arguments: {', '.join(sorted(unknown))}")
                payload = schema.model_validate(arguments).model_dump()
            else:
                payload = arguments
            result = handler(**payload)
            return await result if inspect.isawaitable(result) else result

        return await asyncio.wait_for(execute(), timeout=timeout) if timeout else await execute()
