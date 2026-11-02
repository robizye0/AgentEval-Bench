from agenteval.core.schema import ToolCall
from agenteval.evaluators.trajectory import TrajectoryEvaluator


def test_missing_expected_trajectory_has_zero_efficiency():
    result = TrajectoryEvaluator().evaluate([], optimal_steps=2)
    assert result.efficiency == 0.0


def test_empty_optimal_trajectory_is_fully_efficient():
    result = TrajectoryEvaluator().evaluate([], optimal_steps=0)
    assert result.efficiency == 1.0


def test_repeated_different_calls_do_not_trigger_loop():
    calls = [ToolCall(name="search", kwargs={"query": str(i)}) for i in range(3)]
    assert not TrajectoryEvaluator().evaluate(calls, optimal_steps=2).loop_detected
