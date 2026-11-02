"""Pytest gate for offline AgentEval suites."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import EvaluationSuite, TrajectoryStep


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agent-eval")
    group.addoption(
        "--agent-eval-gate",
        action="store_true",
        default=False,
        help="fail session when the configured agent evaluation suite misses its gate",
    )
    group.addoption("--agent-eval-config", default="eval_suite.yaml", help="suite YAML path")
    group.addoption(
        "--accuracy-threshold", type=float, default=None, help="override suite accuracy threshold"
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("--agent-eval-gate"):
        return
    path = Path(session.config.getoption("--agent-eval-config"))
    try:
        suite = EvaluationSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

        async def evaluate_cases():
            results = []
            for case in suite.cases:

                def replay(prompt: str, recorded=case) -> TrajectoryStep:
                    return TrajectoryStep(
                        input_state=prompt,
                        model_response="recorded",
                        tool_calls=recorded.observed_tool_calls,
                        token_count=recorded.observed_token_count,
                        cost_usd=recorded.observed_cost_usd,
                        latency_ms=recorded.observed_latency_ms,
                    )

                results.append(await EvaluationRunner(replay).run_case(case))
            return results

        results = asyncio.run(evaluate_cases())
        accuracy = sum(result.tool_accuracy_score for result in results) / max(len(results), 1)
        threshold = session.config.getoption("--accuracy-threshold")
        if threshold is None:
            threshold = suite.accuracy_threshold
        if accuracy < threshold or any(not result.passed for result in results):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            terminal = session.config.pluginmanager.get_plugin("terminalreporter")
            if terminal:
                terminal.write_sep(
                    "=", f"AgentEval gate failed: accuracy {accuracy:.3f} < {threshold:.3f}"
                )
    except Exception as exc:  # noqa: BLE001
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal:
            terminal.write_sep("=", f"AgentEval gate error: {exc}")
