import asyncio

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep


def test_runner_hard_fails_token_budget_while_preserving_scores():
    case = TestCase(
        id="budget",
        input_prompt="x",
        expected_tool_sequence=["lookup"],
        max_token_budget=10,
        max_cost_usd=0.01,
    )
    step = TrajectoryStep(
        input_state="x",
        model_response="ok",
        tool_calls=[ToolCall(name="lookup")],
        token_count=11,
        cost_usd=0.005,
    )
    result = asyncio.run(EvaluationRunner(lambda _: step).run_case(case))
    assert result.passed is False
    assert result.details["budget_violations"] == ["token budget exceeded: 11 > 10"]
    assert result.tool_accuracy_score == 1.0


def test_runner_hard_fails_cost_budget():
    case = TestCase(
        id="cost-budget",
        input_prompt="x",
        expected_tool_sequence=["lookup"],
        max_cost_usd=0.001,
    )
    step = TrajectoryStep(
        input_state="x",
        model_response="ok",
        tool_calls=[ToolCall(name="lookup")],
        cost_usd=0.002,
    )
    result = asyncio.run(EvaluationRunner(lambda _: step).run_case(case))
    assert result.passed is False
    assert result.details["budget_violations"] == ["cost budget exceeded: 0.002000 > 0.001000 USD"]
