import asyncio

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep


class TraceCollector:
    def __init__(self):
        self.records = []
        self.calls = []

    def record_result(
        self,
        case_id,
        passed,
        accuracy,
        latency_ms,
        cost_usd,
        token_count=0,
        semantic_intent_score=None,
    ):
        self.records.append(
            (case_id, passed, accuracy, latency_ms, cost_usd, token_count, semantic_intent_score)
        )

    def record_tool_call(self, case_id, name, latency_ms):
        self.calls.append((case_id, name, latency_ms))


def test_runner_emits_result_latency_and_cost_to_configured_tracer():
    case = TestCase(id="trace", input_prompt="x", expected_tool_sequence=["fetch"])
    step = TrajectoryStep(
        input_state="x",
        model_response="done",
        tool_calls=[ToolCall(name="fetch", latency_ms=23)],
        token_count=10,
        cost_usd=0.004,
    )
    tracer = TraceCollector()
    result = asyncio.run(EvaluationRunner(lambda _: step, tracer=tracer).run_case(case))
    assert result.passed
    assert tracer.records == [("trace", True, 1.0, 23, 0.004, 10, None)]
    assert tracer.calls == [("trace", "fetch", 23)]
