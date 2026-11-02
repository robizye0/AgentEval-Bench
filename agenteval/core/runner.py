"""Async, adapter-neutral evaluation runner."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, create_model

from agenteval.core.schema import EvalResult, TestCase, TrajectoryStep
from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
from agenteval.evaluators.trajectory import TrajectoryEvaluator

AgentAdapter = Callable[[str], Awaitable[TrajectoryStep] | TrajectoryStep]


def _model_from_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    fields = {key: (Any, ...) for key in schema.get("required", [])}
    props = schema.get("properties", {})
    types = {"string": str, "integer": int, "number": float, "boolean": bool}
    for key, spec in props.items():
        typ = types.get(spec.get("type"), Any)
        fields.setdefault(key, (typ, ... if key in schema.get("required", []) else None))
    return create_model(name, __config__={"extra": "forbid"}, **fields)


class EvaluationRunner:
    def __init__(self, adapter: AgentAdapter) -> None:
        self.adapter = adapter

    async def run_case(self, case: TestCase) -> EvalResult:
        try:
            outcome = self.adapter(case.input_prompt)
            step = await outcome if inspect.isawaitable(outcome) else outcome
            if not isinstance(step, TrajectoryStep):
                raise TypeError("agent adapter must return TrajectoryStep")
        except Exception as exc:  # noqa: BLE001
            return EvalResult(
                passed=False,
                tool_accuracy_score=0,
                trajectory_efficiency=0,
                resilience_score=0,
                schema_errors=[f"agent crashed: {type(exc).__name__}: {exc}"],
            )
        schemas = {
            name: _model_from_schema(name.title().replace("_", "") + "Args", schema)
            for name, schema in case.argument_schemas.items()
        }
        accuracy = ToolAccuracyEvaluator().evaluate(case, step.tool_calls, schemas)
        trajectory = TrajectoryEvaluator().evaluate(
            step.tool_calls,
            case.optimal_steps
            if case.optimal_steps is not None
            else len(case.expected_tool_sequence),
        )
        errors = list(accuracy.schema_errors)
        if trajectory.loop_detected:
            errors.append("repeated identical tool call loop detected")
        if len(step.tool_calls) > case.max_allowed_steps:
            errors.append(f"step limit exceeded: {len(step.tool_calls)} > {case.max_allowed_steps}")
        passed = accuracy.score >= 1.0 and trajectory.efficiency >= 1.0 and not errors
        return EvalResult(
            passed=passed,
            tool_accuracy_score=accuracy.score,
            trajectory_efficiency=trajectory.efficiency,
            schema_errors=errors,
            total_cost_usd=step.cost_usd,
            details={"case_id": case.id, "token_count": step.token_count},
        )

    async def run(self, cases: list[TestCase]) -> list[EvalResult]:
        import asyncio

        return list(await asyncio.gather(*(self.run_case(case) for case in cases)))
