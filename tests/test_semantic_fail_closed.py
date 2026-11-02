import asyncio

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep
from agenteval.evaluators.semantic_intent import IntentScore, SemanticIntentEvaluator


class AlwaysApproveJudge:
    async def score_intent(self, expected, actual):
        return IntentScore(score=1.0, rationale="test judge deliberately returns maximum")


def test_semantic_judge_cannot_override_unmarked_deterministic_fields():
    case = TestCase(
        id="account-lookup",
        input_prompt="look up account 123",
        expected_tool_sequence=["account_lookup"],
        required_arguments={"account_id": "123"},
    )
    step = TrajectoryStep(
        input_state=case.input_prompt,
        model_response="done",
        tool_calls=[ToolCall(name="account_lookup", kwargs={"account_id": "999"})],
    )
    result = asyncio.run(
        EvaluationRunner(
            lambda _: step,
            semantic_judge=SemanticIntentEvaluator(AlwaysApproveJudge()),
        ).run_case(case)
    )
    assert result.semantic_intent_score == 1.0
    assert result.passed is False
    assert result.details["argument_errors"]
