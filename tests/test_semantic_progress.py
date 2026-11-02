import asyncio

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep
from agenteval.evaluators.semantic_intent import IntentScore, SemanticIntentEvaluator


class ProgressJudge:
    async def score_intent(self, expected, actual):
        return IntentScore(score=0.91, rationale="extra step enriches the valid path")


def test_semantic_judge_accepts_declared_additional_progress_step():
    case = TestCase(
        id="progress",
        input_prompt="check weather and nearby cafes",
        expected_tool_sequence=["weather", "places"],
        allowed_additional_tools=["history"],
    )
    step = TrajectoryStep(
        input_state="",
        model_response="complete",
        tool_calls=[ToolCall(name="places"), ToolCall(name="history"), ToolCall(name="weather")],
    )
    result = asyncio.run(
        EvaluationRunner(
            lambda _: step, semantic_judge=SemanticIntentEvaluator(ProgressJudge())
        ).run_case(case)
    )
    assert result.trajectory_efficiency < 1.0
    assert result.passed is True
