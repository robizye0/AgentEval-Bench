from pathlib import Path

from typer.testing import CliRunner

from agenteval.cli import app


def test_cli_writes_html_report(tmp_path: Path):
    config = tmp_path / "suite.json"
    config.write_text(
        '{"cases":[{"id":"report-case","input_prompt":"x","expected_tool_sequence":["find"],'
        '"observed_tool_calls":[{"name":"find","kwargs":{}}],"observed_token_count":7,"observed_latency_ms":11}]}',
        encoding="utf-8",
    )
    report = tmp_path / "summary.html"
    result = CliRunner().invoke(app, ["run", "--config", str(config), "--html-report", str(report)])
    assert result.exit_code == 0, result.output
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "report-case" in html
    assert "accuracy-chart" in html
