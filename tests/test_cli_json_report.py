import json
from pathlib import Path

from typer.testing import CliRunner

from agenteval.cli import app


def test_cli_writes_machine_readable_json_summary_even_when_suite_fails(tmp_path: Path):
    config = tmp_path / "suite.json"
    config.write_text(
        '{"cases":[{"id":"over-budget","input_prompt":"x","expected_tool_sequence":["search"],'
        '"observed_tool_calls":[{"name":"search","kwargs":{}}],"observed_token_count":11,"max_token_budget":10}]}',
        encoding="utf-8",
    )
    output = tmp_path / "results" / "eval.json"
    result = CliRunner().invoke(
        app, ["run", "--config", str(config), "--json-report", str(output)]
    )
    assert result.exit_code == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert summary["cases"][0]["details"]["budget_violations"]
