"""Trajectory efficiency and repeated-call loop detection."""

from __future__ import annotations

import json
from collections import Counter

from pydantic import BaseModel, Field

from agenteval.core.schema import ToolCall


class TrajectoryResult(BaseModel):
    efficiency: float = Field(ge=0, le=1)
    loop_detected: bool
    repeated_calls: list[str]
    actual_steps: int


class TrajectoryEvaluator:
    def __init__(self, repeat_limit: int = 3) -> None:
        if repeat_limit < 2:
            raise ValueError("repeat_limit must be at least 2")
        self.repeat_limit = repeat_limit

    def evaluate(self, calls: list[ToolCall], optimal_steps: int) -> TrajectoryResult:
        if optimal_steps < 0:
            raise ValueError("optimal_steps must be non-negative")
        keys = [
            f"{call.name}:{json.dumps(call.kwargs, sort_keys=True, default=str)}" for call in calls
        ]
        repeated = sorted(key for key, count in Counter(keys).items() if count >= self.repeat_limit)
        actual = len(calls)
        efficiency = (
            1.0
            if actual == 0 and optimal_steps == 0
            else (0.0 if actual == 0 else min(1.0, optimal_steps / actual))
        )
        return TrajectoryResult(
            efficiency=efficiency,
            loop_detected=bool(repeated),
            repeated_calls=repeated,
            actual_steps=actual,
        )
