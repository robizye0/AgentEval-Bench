from pathlib import Path

from typer.testing import CliRunner

from agenteval.cli import app


def test_cli_replays_distinct_cases_with_same_prompt(tmp_path: Path):
    config = tmp_path / "same-prompt.json"
    config.write_text(
        '{"cases":['
        '{"id":"search","input_prompt":"same","expected_tool_sequence":["search"],'
        '"observed_tool_calls":[{"name":"search","kwargs":{}}]},'
        '{"id":"weather","input_prompt":"same","expected_tool_sequence":["weather"],'
        '"observed_tool_calls":[{"name":"weather","kwargs":{}}]}]}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "search" in result.output
    assert "weather" in result.output
