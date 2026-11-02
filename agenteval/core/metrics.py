"""Metrics helpers."""

from agenteval.core.schema import EvalResult


def aggregate_pass_rate(results: list[EvalResult]) -> float:
    return sum(result.passed for result in results) / len(results) if results else 0.0
