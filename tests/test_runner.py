import asyncio

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep


def test_runner_preserves_adapter_reported_cost():
    case = TestCase(id="cost", input_prompt="run", expected_tool_sequence=["search"])
    step = TrajectoryStep(
        input_state="run",
        model_response="done",
        tool_calls=[ToolCall(name="search")],
        cost_usd=0.012,
    )
    result = asyncio.run(EvaluationRunner(lambda _: step).run_case(case))
    assert result.total_cost_usd == 0.012
