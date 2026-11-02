import asyncio

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep
from agenteval.evaluators.semantic_intent import IntentScore, SemanticIntentEvaluator


class FixedJudge:
    def __init__(self, score: float):
        self.value = score

    async def score_intent(self, expected, actual):
        return IntentScore(score=self.value, rationale="semantic match")


def test_semantic_judge_accepts_equivalent_search_query():
    case = TestCase(
        id="search",
        input_prompt="Find coffee shops",
        expected_tool_sequence=["search"],
        semantic_arguments=["query"],
        required_arguments={"query": "best coffee in London"},
    )
    step = TrajectoryStep(
        input_state=case.input_prompt,
        model_response="done",
        tool_calls=[ToolCall(name="search", kwargs={"query": "top rated coffee shops in London"})],
    )
    runner = EvaluationRunner(
        lambda _: step, semantic_judge=SemanticIntentEvaluator(FixedJudge(0.94))
    )
    result = asyncio.run(runner.run_case(case))
    assert result.deterministic_accuracy_score == 0.0
    assert result.semantic_intent_score == 0.94
    assert result.passed is True


def test_semantic_judge_allows_reordered_progress_but_not_tool_substitution():
    case = TestCase(
        id="trajectory",
        input_prompt="check weather and nearby cafes",
        expected_tool_sequence=["weather", "places"],
    )
    valid = TrajectoryStep(
        input_state="",
        model_response="done",
        tool_calls=[ToolCall(name="places"), ToolCall(name="weather")],
    )
    result = asyncio.run(
        EvaluationRunner(
            lambda _: valid, semantic_judge=SemanticIntentEvaluator(FixedJudge(0.9))
        ).run_case(case)
    )
    assert result.passed is True
    wrong = TrajectoryStep(
        input_state="",
        model_response="done",
        tool_calls=[ToolCall(name="places"), ToolCall(name="sql_delete")],
    )
    failed = asyncio.run(
        EvaluationRunner(
            lambda _: wrong, semantic_judge=SemanticIntentEvaluator(FixedJudge(1.0))
        ).run_case(case)
    )
    assert failed.passed is False
