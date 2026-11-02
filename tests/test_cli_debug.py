from pathlib import Path

from typer.testing import CliRunner

from agenteval.cli import app


def test_debug_command_steps_backward_and_shows_validation_errors_and_tool_responses(tmp_path: Path):
    config = tmp_path / "recorded.json"
    config.write_text(
        '{"cases":[{"id":"debug-me","input_prompt":"Find a fact",'
        '"expected_tool_sequence":["search","lookup"],'
        '"observed_tool_calls":[{"name":"search","kwargs":{"query":"fact"}},'
        '{"name":"lookup","kwargs":{"count":"many"}}],'
        '"observed_tool_responses":[{"hits":2},"mock lookup result"],'
        '"argument_schemas":{"search":{"type":"object","required":["query"],'
        '"properties":{"query":{"type":"string"}}},'
        '"lookup":{"type":"object","required":["count"],'
        '"properties":{"count":{"type":"integer"}}}}}]}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["debug", "--case-id", "debug-me", "--config", str(config)],
        input="n\np\nq\n",
    )
    assert result.exit_code == 0, result.output
    assert "Turn: 1 / 2" in result.output
    assert "Turn: 2 / 2" in result.output
    assert "Turn: 1 / 2" in result.output[result.output.index("Turn: 2 / 2") :]
    assert "mock lookup result" in result.output
    assert "valid integer" in result.output, result.output
