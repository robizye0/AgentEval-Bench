"""Tool selection, payload validation, and precision scoring."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from agenteval.core.schema import TestCase, ToolCall


class ToolAccuracyResult(BaseModel):
    score: float
    schema_errors: list[str]
    expected: list[str]
    actual: list[str]


class ToolAccuracyEvaluator:
    def evaluate(
        self,
        case: TestCase,
        calls: list[ToolCall],
        schemas: dict[str, type[BaseModel]] | None = None,
    ) -> ToolAccuracyResult:
        schemas = schemas or {}
        expected, actual = case.expected_tool_sequence, [call.name for call in calls]
        errors: list[str] = []
        matched = sum(
            1 for i, name in enumerate(actual) if i < len(expected) and name == expected[i]
        )
        # Each expected call is one opportunity; extra calls reduce precision too.
        sequence_score = matched / max(len(expected), len(actual), 1)
        valid = 0
        for index, call in enumerate(calls):
            schema = schemas.get(call.name)
            try:
                if schema:
                    unknown = set(call.kwargs) - set(schema.model_fields)
                    if unknown:
                        errors.append(
                            f"{call.name}: unexpected arguments: {', '.join(sorted(unknown))}"
                        )
                        continue
                    schema.model_validate(call.kwargs)
            except ValidationError as exc:
                errors.extend(f"{call.name}: {e['loc']}: {e['msg']}" for e in exc.errors())
                continue
            required = case.required_arguments if index == 0 and call.name in expected else {}
            if any(call.kwargs.get(key) != value for key, value in required.items()):
                errors.append(f"{call.name}: required arguments do not match expected values")
                continue
            valid += 1
        payload_score = valid / max(len(calls), 1) if calls else (1.0 if not expected else 0.0)
        return ToolAccuracyResult(
            score=max(0.0, min(1.0, sequence_score * payload_score)),
            schema_errors=errors,
            expected=expected,
            actual=actual,
        )
