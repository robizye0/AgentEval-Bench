from agenteval.core.schema import EvalResult
from agenteval.reporting.html import render_html_report


def test_html_report_is_self_contained_and_shows_accuracy_latency_and_tokens():
    report = render_html_report(
        [
            EvalResult(
                passed=False,
                tool_accuracy_score=0.5,
                deterministic_accuracy_score=0.5,
                trajectory_efficiency=0.8,
                schema_errors=["wrong tool"],
                details={"case_id": "case-a", "token_count": 42, "latency_ms": 3000},
            )
        ]
    )
    assert 'id="accuracy-chart"' in report
    assert 'id="latency-chart"' in report
    assert 'id="token-chart"' in report
    assert "case-a" in report
    assert "wrong tool" in report
    assert "https://" not in report
    assert "<script>" in report
