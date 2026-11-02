"""Strict, serializable models used throughout AgentEval-Bench."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ToolCall(StrictModel):
    name: str = Field(min_length=1)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    parsed_payload: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def payload_consistent(self) -> ToolCall:
        if self.parsed_payload is not None and self.parsed_payload != self.kwargs:
            raise ValueError("parsed_payload must match kwargs when both are supplied")
        return self


class TrajectoryStep(StrictModel):
    input_state: str
    model_response: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_responses: list[Any] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)


class EvalResult(StrictModel):
    passed: bool
    tool_accuracy_score: float = Field(ge=0, le=1)
    deterministic_accuracy_score: float = Field(default=0.0, ge=0, le=1)
    semantic_intent_score: float | None = Field(default=None, ge=0, le=1)
    trajectory_efficiency: float = Field(ge=0, le=1)
    schema_errors: list[str] = Field(default_factory=list)
    total_cost_usd: float = Field(default=0, ge=0)
    resilience_score: float = Field(default=1.0, ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)


class TestCase(StrictModel):
    __test__ = False
    id: str = Field(default="case", min_length=1)
    input_prompt: str
    expected_tool_sequence: list[str]
    allowed_additional_tools: list[str] = Field(default_factory=list)
    semantic_arguments: list[str] = Field(default_factory=list)
    required_arguments: dict[str, Any] = Field(default_factory=dict)
    max_allowed_steps: int = Field(default=10, gt=0)
    max_token_budget: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    optimal_steps: int | None = Field(default=None, ge=0)
    argument_schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    expected_output: Any | None = None
    failure_scenario: str | None = None
    observed_tool_calls: list[ToolCall] = Field(default_factory=list)
    observed_token_count: int = Field(default=0, ge=0)
    observed_cost_usd: float = Field(default=0, ge=0)
    observed_latency_ms: float = Field(default=0, ge=0)


class EvaluationSuite(StrictModel):
    cases: list[TestCase]
    accuracy_threshold: float = Field(default=0.95, ge=0, le=1)
    max_steps: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def unique_ids(self) -> EvaluationSuite:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("test case ids must be unique")
        return self
