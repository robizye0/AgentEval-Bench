from pathlib import Path

from typer.testing import CliRunner

from agenteval.cli import app


def test_cli_returns_success_for_perfect_fixture(tmp_path: Path):
    config = tmp_path / "suite.json"
    config.write_text(
        '{"accuracy_threshold":1,"cases":[{"id":"ok","input_prompt":"x","expected_tool_sequence":["find"],"observed_tool_calls":[{"name":"find","kwargs":{}}]}]}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert '"passed": true' in result.output


def test_cli_returns_gate_failure_for_wrong_tool(tmp_path: Path):
    config = tmp_path / "suite.json"
    config.write_text(
        '{"accuracy_threshold":0.5,"cases":[{"id":"wrong","input_prompt":"x","expected_tool_sequence":["find"],"observed_tool_calls":[{"name":"other","kwargs":{}}]}]}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 1
