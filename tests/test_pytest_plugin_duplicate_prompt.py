import json
import subprocess
import sys


def test_pytest_gate_replays_duplicate_prompts_by_case(tmp_path):
    suite = {
        "cases": [
            {
                "id": "search",
                "input_prompt": "same",
                "expected_tool_sequence": ["search"],
                "observed_tool_calls": [{"name": "search", "kwargs": {}}],
            },
            {
                "id": "weather",
                "input_prompt": "same",
                "expected_tool_sequence": ["weather"],
                "observed_tool_calls": [{"name": "weather", "kwargs": {}}],
            },
        ]
    }
    config = tmp_path / "suite.json"
    config.write_text(json.dumps(suite), encoding="utf-8")
    test_file = tmp_path / "test_placeholder.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--agent-eval-gate",
            "--agent-eval-config",
            str(config),
            str(test_file),
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
