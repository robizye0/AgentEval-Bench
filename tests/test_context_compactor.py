from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep
from agenteval.evaluators.context_compactor import ContextCompactorEvaluator, ContextSegment


def test_compactor_detects_html_bloat_and_returns_exact_reduction_strategy():
    raw_html = (
        "<html><style>body { color: red; margin: 0; }" * 80
        + "</style><body><h1>Relevant report</h1><p>Revenue increased 14 percent.</p></body></html>"
    )
    report = ContextCompactorEvaluator().analyze(
        [ContextSegment(kind="tool_response", name="search_web", content=raw_html)]
    )
    assert report.suggestions
    suggestion = report.suggestions[0]
    assert suggestion.segment == "search_web"
    assert suggestion.noise_ratio > 0.5
    assert "<style>" in suggestion.strategy
    assert suggestion.estimated_savings_tokens > 0


def test_runner_attaches_compaction_diagnostics_on_budget_breach():
    case = TestCase(
        id="over-budget-html",
        input_prompt="Summarize this page",
        expected_tool_sequence=["search_web"],
        max_token_budget=10,
    )
    step = TrajectoryStep(
        input_state=case.input_prompt,
        model_response="finished",
        tool_calls=[ToolCall(name="search_web")],
        tool_responses=["<style>" + ("body{color:red}" * 500) + "</style>Relevant text"],
        token_count=20,
    )
    result = __import__("asyncio").run(EvaluationRunner(lambda _: step).run_case(case))
    assert not result.passed
    assert result.details["compaction_suggestions"]
    assert result.details["compaction_suggestions"][0]["estimated_savings_tokens"] > 0
