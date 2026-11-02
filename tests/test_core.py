import pytest
from pydantic import BaseModel

from agenteval.core.schema import EvalResult, TestCase, ToolCall, TrajectoryStep
from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
from agenteval.evaluators.trajectory import TrajectoryEvaluator


class SearchArgs(BaseModel):
    query: str
    limit: int


def test_test_case_rejects_non_positive_step_limit():
    with pytest.raises(ValueError):
        TestCase(input_prompt="find it", expected_tool_sequence=["search"], max_allowed_steps=0)


def test_tool_accuracy_rejects_extra_arguments_and_mismatched_sequence():
    case = TestCase(
        input_prompt="search",
        expected_tool_sequence=["search"],
        required_arguments={"query": "cats", "limit": 2},
    )
    calls = [ToolCall(name="search", kwargs={"query": "cats", "limit": 2, "noise": True})]
    result = ToolAccuracyEvaluator().evaluate(case, calls, schemas={"search": SearchArgs})
    assert result.score == 0.0
    assert result.schema_errors


def test_trajectory_detects_identical_retries_three_times():
    calls = [ToolCall(name="search", kwargs={"query": "x"}) for _ in range(3)]
    result = TrajectoryEvaluator().evaluate(calls, optimal_steps=1)
    assert result.loop_detected is True
    assert result.efficiency == pytest.approx(1 / 3)


def test_eval_result_enforces_bounded_scores():
    with pytest.raises(ValueError):
        EvalResult(passed=False, tool_accuracy_score=1.1, trajectory_efficiency=0.0)


def test_trajectory_step_accepts_recorded_responses():
    step = TrajectoryStep(
        input_state="hello",
        model_response="search now",
        tool_calls=[],
        tool_responses=[],
        token_count=4,
    )
    assert step.token_count == 4
