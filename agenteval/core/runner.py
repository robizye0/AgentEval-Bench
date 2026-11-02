"""Async, adapter-neutral evaluation runner."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, create_model

from agenteval.core.schema import EvalResult, TestCase, TrajectoryStep
from agenteval.evaluators.semantic_intent import SemanticIntentEvaluator
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
    def __init__(
        self,
        adapter: AgentAdapter,
        semantic_judge: SemanticIntentEvaluator | None = None,
        semantic_threshold: float = 0.8,
        tracer: Any | None = None,
    ) -> None:
        if not 0 <= semantic_threshold <= 1:
            raise ValueError("semantic_threshold must be between 0 and 1")
        self.adapter = adapter
        self.semantic_judge = semantic_judge
        self.semantic_threshold = semantic_threshold
        self.tracer = tracer

    def _record_trace(self, result: EvalResult) -> None:
        if self.tracer is None:
            return
        latency_ms = float(result.details.get("latency_ms", 0))
        self.tracer.record_result(
            str(result.details.get("case_id", "case")),
            result.passed,
            result.deterministic_accuracy_score,
            latency_ms,
            result.total_cost_usd,
            token_count=int(result.details.get("token_count", 0)),
            semantic_intent_score=result.semantic_intent_score,
        )
        record_tool_call = getattr(self.tracer, "record_tool_call", None)
        if record_tool_call is not None:
            for call in result.details.get("tool_calls", []):
                record_tool_call(
                    str(result.details.get("case_id", "case")),
                    str(call["name"]),
                    float(call["latency_ms"]),
                )

    async def run_case(self, case: TestCase) -> EvalResult:
        try:
            outcome = self.adapter(case.input_prompt)
            step = await outcome if inspect.isawaitable(outcome) else outcome
            if not isinstance(step, TrajectoryStep):
                raise TypeError("agent adapter must return TrajectoryStep")
        except Exception as exc:  # noqa: BLE001
            result = EvalResult(
                passed=False,
                tool_accuracy_score=0,
                deterministic_accuracy_score=0,
                trajectory_efficiency=0,
                resilience_score=0,
                schema_errors=[f"agent crashed: {type(exc).__name__}: {exc}"],
                details={"case_id": case.id},
            )
            self._record_trace(result)
            return result
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
        budget_violations: list[str] = []
        if case.max_token_budget is not None and step.token_count > case.max_token_budget:
            budget_violations.append(
                f"token budget exceeded: {step.token_count} > {case.max_token_budget}"
            )
        if case.max_cost_usd is not None and step.cost_usd > case.max_cost_usd:
            budget_violations.append(
                f"cost budget exceeded: {step.cost_usd:.6f} > {case.max_cost_usd:.6f} USD"
            )
        semantic_score: float | None = None
        semantic_rationale: str | None = None
        if self.semantic_judge is not None:
            try:
                semantic = await self.semantic_judge.evaluate(case, step.tool_calls)
                semantic_score = semantic.score
                semantic_rationale = semantic.rationale
            except Exception as exc:  # noqa: BLE001
                errors.append(f"semantic judge failed: {type(exc).__name__}: {exc}")
        expected_tools = Counter(case.expected_tool_sequence)
        observed_tools = Counter(call.name for call in step.tool_calls)
        if self.semantic_judge is None:
            accuracy_gate = accuracy.score >= 1.0
            trajectory_gate = trajectory.efficiency >= 1.0
        else:
            extra_tools = observed_tools - expected_tools
            allowed_tools = Counter(case.allowed_additional_tools)
            extra_tools_allowed = all(
                count <= allowed_tools[name] for name, count in extra_tools.items()
            )
            accuracy_gate = (
                semantic_score is not None
                and semantic_score >= self.semantic_threshold
                and accuracy.schema_valid
                and set(accuracy.argument_mismatch_fields) <= set(case.semantic_arguments)
                and expected_tools <= observed_tools
                and extra_tools_allowed
            )
            trajectory_gate = accuracy_gate
        passed = accuracy_gate and trajectory_gate and not errors and not budget_violations
        result = EvalResult(
            passed=passed,
            tool_accuracy_score=accuracy.score,
            deterministic_accuracy_score=accuracy.score,
            semantic_intent_score=semantic_score,
            trajectory_efficiency=trajectory.efficiency,
            schema_errors=errors,
            total_cost_usd=step.cost_usd,
            details={
                "case_id": case.id,
                "expected_tools": case.expected_tool_sequence,
                "observed_tools": [call.name for call in step.tool_calls],
                "tool_calls": [
                    {"name": call.name, "latency_ms": call.latency_ms} for call in step.tool_calls
                ],
                "token_count": step.token_count,
                "latency_ms": step.latency_ms or sum(call.latency_ms for call in step.tool_calls),
                "argument_errors": accuracy.argument_errors,
                "budget_violations": budget_violations,
                "semantic_rationale": semantic_rationale,
            },
        )
        self._record_trace(result)
        return result

    async def run(self, cases: list[TestCase]) -> list[EvalResult]:
        import asyncio

        return list(await asyncio.gather(*(self.run_case(case) for case in cases)))
