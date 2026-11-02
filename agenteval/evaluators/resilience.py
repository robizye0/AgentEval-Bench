"""Fault injection and graceful-failure scoring."""

from __future__ import annotations

import asyncio
import inspect
import random
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field


class InjectedFailure(Exception):
    """Raised when a configured synthetic tool failure is triggered."""

    def __init__(self, scenario: str):
        self.scenario = scenario
        super().__init__(f"simulated {scenario} failure")


class ResilienceResult(BaseModel):
    score: float = Field(ge=0, le=1)
    recovered: bool
    error: str | None = None


class ResilienceEvaluator:
    """Run an agent harness with a failure callback it can trigger and recover from."""

    async def evaluate(
        self, operation: Callable[..., Awaitable[Any]], scenario: str
    ) -> ResilienceResult:
        self.validate_scenario(scenario)

        def inject() -> None:
            raise InjectedFailure(scenario)

        try:
            result = (
                await operation(inject)
                if len(inspect.signature(operation).parameters)
                else await operation()
            )
        except Exception as exc:  # noqa: BLE001
            return ResilienceResult(
                score=0.0, recovered=False, error=f"{type(exc).__name__}: {exc}"
            )
        if isinstance(result, dict) and result.get("recovered") is False:
            return ResilienceResult(
                score=0.0,
                recovered=False,
                error=str(result.get("error", "agent reported unrecovered failure")),
            )
        return ResilienceResult(score=1.0, recovered=True)

    @staticmethod
    def calculate_delay_ms(
        base_latency_ms: float = 0,
        jitter_ms: int = 0,
        spike_probability: float = 0,
        spike_latency_ms: float = 3000,
        seed: int = 0,
    ) -> float:
        if base_latency_ms < 0 or jitter_ms < 0 or spike_latency_ms < 0:
            raise ValueError("latency values must be non-negative")
        if not 0 <= spike_probability <= 1:
            raise ValueError("spike_probability must be between 0 and 1")
        rng = random.Random(seed)
        delay = max(0.0, base_latency_ms + rng.randint(-jitter_ms, jitter_ms))
        if rng.random() < spike_probability:
            delay += spike_latency_ms
        return delay

    @classmethod
    async def inject_latency(cls, **profile: Any) -> float:
        delay_ms = cls.calculate_delay_ms(**profile)
        await asyncio.sleep(delay_ms / 1000)
        return delay_ms

    @staticmethod
    async def stream_with_chaos(
        chunks: Iterable[bytes] | AsyncIterable[bytes], drop_after_chunks: int
    ) -> AsyncIterator[bytes]:
        if drop_after_chunks < 0:
            raise ValueError("drop_after_chunks must be non-negative")
        count = 0
        if isinstance(chunks, AsyncIterable):
            async for chunk in chunks:
                if count >= drop_after_chunks:
                    raise ConnectionResetError("simulated partial HTTP stream drop")
                yield chunk
                count += 1
            return
        for chunk in chunks:
            if count >= drop_after_chunks:
                raise ConnectionResetError("simulated partial HTTP stream drop")
            yield chunk
            count += 1

    @staticmethod
    def validate_scenario(scenario: str) -> None:
        if scenario not in {
            "http_500",
            "rate_limit",
            "malformed_json",
            "timeout",
            "schema_validation",
            "connection_reset",
            "partial_stream",
        }:
            raise ValueError(f"unsupported failure scenario: {scenario}")

    @staticmethod
    def inject(scenario: str) -> None:
        ResilienceEvaluator.validate_scenario(scenario)
        if scenario == "connection_reset":
            raise ConnectionResetError("simulated connection reset")
        raise InjectedFailure(scenario)
