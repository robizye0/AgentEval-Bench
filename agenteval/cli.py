"""Command-line entry point for deterministic suite evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import EvaluationSuite, TrajectoryStep

app = typer.Typer(help="Evaluate deterministic agent trajectories in CI.")


@app.callback()
def main() -> None:
    """AgentEval-Bench evaluation commands."""


@app.command("run")
def run_suite(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),  # noqa: B008
    accuracy_threshold: float | None = typer.Option(None, min=0, max=1),
) -> None:
    """Run cases containing recorded `observed_tool_calls` against expected trajectories."""
    try:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        suite = EvaluationSuite.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        typer.echo(f"Invalid evaluation suite: {exc}", err=True)
        raise typer.Exit(code=2)
    threshold = suite.accuracy_threshold if accuracy_threshold is None else accuracy_threshold
    output = []
    for case in suite.cases:
        step = TrajectoryStep(
            input_state=case.input_prompt,
            model_response="recorded evaluation",
            tool_calls=case.observed_tool_calls,
        )
        result = asyncio.run(EvaluationRunner(lambda _, recorded=step: recorded).run_case(case))
        output.append({"id": case.id, **result.model_dump()})
    avg = sum(item["tool_accuracy_score"] for item in output) / max(len(output), 1)
    report = {
        "cases": output,
        "average_tool_accuracy": avg,
        "threshold": threshold,
        "passed": avg >= threshold and all(item["passed"] for item in output),
    }
    typer.echo(json.dumps(report, indent=2, default=str))
    if not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
