"""Fault injection and graceful-failure scoring."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
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
    def validate_scenario(scenario: str) -> None:
        if scenario not in {
            "http_500",
            "rate_limit",
            "malformed_json",
            "timeout",
            "schema_validation",
        }:
            raise ValueError(f"unsupported failure scenario: {scenario}")

    @staticmethod
    def inject(scenario: str) -> None:
        ResilienceEvaluator.validate_scenario(scenario)
        raise InjectedFailure(scenario)
