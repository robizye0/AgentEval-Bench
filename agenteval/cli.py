"""Command-line entry point for deterministic suite evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import EvaluationSuite, TrajectoryStep
from agenteval.evaluators.semantic_intent import LLMIntentJudge, SemanticIntentEvaluator
from agenteval.integration.langfuse_tracer import EvaluationTracer
from agenteval.reporting.html import write_html_report
from agenteval.reporting.terminal import render_results

app = typer.Typer(help="Evaluate deterministic agent trajectories in CI.")


@app.callback()
def main() -> None:
    """AgentEval-Bench evaluation commands."""


@app.command("run")
def run_suite(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),  # noqa: B008
    accuracy_threshold: float | None = typer.Option(None, min=0, max=1),
    semantic_judge: bool = typer.Option(False, "--semantic-judge"),
    judge_provider: str = typer.Option("openai", "--judge-provider"),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    semantic_threshold: float = typer.Option(0.8, "--semantic-threshold", min=0, max=1),
    html_report: Path | None = typer.Option(None, "--html-report"),  # noqa: B008
) -> None:
    """Evaluate recorded trajectories; optionally call an LLM judge and write HTML."""
    console = Console(stderr=True)
    try:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        suite = EvaluationSuite.model_validate(raw)
        judge = (
            SemanticIntentEvaluator(LLMIntentJudge(provider=judge_provider, model=judge_model))
            if semantic_judge
            else None
        )
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        console.print(f"[red]Invalid evaluation configuration:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    threshold = suite.accuracy_threshold if accuracy_threshold is None else accuracy_threshold

    def replay(case):
        def adapter(prompt: str) -> TrajectoryStep:
            return TrajectoryStep(
                input_state=prompt,
                model_response="recorded evaluation",
                tool_calls=case.observed_tool_calls,
                token_count=case.observed_token_count,
                cost_usd=case.observed_cost_usd,
                latency_ms=case.observed_latency_ms,
            )

        return adapter

    tracer = EvaluationTracer()
    results = []
    columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn()]
    with Progress(*columns, console=console) as progress:
        task_id = progress.add_task("Evaluating trajectories", total=len(suite.cases))
        for case in suite.cases:
            runner = EvaluationRunner(
                replay(case),
                semantic_judge=judge,
                semantic_threshold=semantic_threshold,
                tracer=tracer,
            )
            result = asyncio.run(runner.run_case(case))
            results.append(result)
            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            progress.update(task_id, advance=1, description=f"{status} {case.id}")
    if not tracer.flush():
        console.print("[yellow]Warning: OpenTelemetry did not flush all evaluation spans.[/yellow]")

    render_results(results, console=console)
    if html_report is not None:
        try:
            write_html_report(html_report, results)
            console.print(f"[cyan]HTML report:[/cyan] {html_report.resolve()}")
        except OSError as exc:
            console.print(f"[red]Could not write HTML report:[/red] {exc}")
            raise typer.Exit(code=2) from exc

    deterministic = [result.deterministic_accuracy_score for result in results]
    semantic = [
        result.semantic_intent_score
        for result in results
        if result.semantic_intent_score is not None
    ]
    effective = [
        result.semantic_intent_score
        if result.semantic_intent_score is not None
        else result.deterministic_accuracy_score
        for result in results
    ]
    average_deterministic = sum(deterministic) / max(len(deterministic), 1)
    average_effective = sum(effective) / max(len(effective), 1)
    report = {
        "cases": [
            {"id": result.details.get("case_id", "case"), **result.model_dump()}
            for result in results
        ],
        "average_tool_accuracy": average_deterministic,
        "average_deterministic_accuracy": average_deterministic,
        "average_semantic_intent_score": sum(semantic) / len(semantic) if semantic else None,
        "average_gate_score": average_effective,
        "threshold": threshold,
        "html_report": str(html_report.resolve()) if html_report else None,
        "passed": average_effective >= threshold and all(result.passed for result in results),
    }
    typer.echo(json.dumps(report, indent=2, default=str))
    if not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
