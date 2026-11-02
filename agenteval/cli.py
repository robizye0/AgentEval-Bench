"""Command-line entry point for deterministic suite evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.text import Text

from agenteval.core.runner import EvaluationRunner, build_argument_models
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
    json_report: Path | None = typer.Option(None, "--json-report"),  # noqa: B008
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
                tool_responses=case.observed_tool_responses,
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
    encoded_report = json.dumps(report, indent=2, default=str)
    if json_report is not None:
        try:
            json_report.parent.mkdir(parents=True, exist_ok=True)
            json_report.write_text(encoded_report, encoding="utf-8")
            console.print(f"[cyan]JSON report:[/cyan] {json_report.resolve()}")
        except OSError as exc:
            console.print(f"[red]Could not write JSON report:[/red] {exc}")
            raise typer.Exit(code=2) from exc
    typer.echo(encoded_report)
    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command("debug")
def debug_case(
    case_id: str = typer.Option(..., "--case-id"),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),  # noqa: B008
    semantic_judge: bool = typer.Option(False, "--semantic-judge"),
    judge_provider: str = typer.Option("openai", "--judge-provider"),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    semantic_threshold: float = typer.Option(0.8, "--semantic-threshold", min=0, max=1),
) -> None:
    """Interactively replay one recorded case, stepping across tool calls."""
    console = Console()
    try:
        suite = EvaluationSuite.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        console.print(f"[red]Invalid evaluation configuration:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    case = next((item for item in suite.cases if item.id == case_id), None)
    if case is None:
        console.print(f"[red]Unknown case id:[/red] {case_id}")
        raise typer.Exit(code=2)

    judge = (
        SemanticIntentEvaluator(LLMIntentJudge(provider=judge_provider, model=judge_model))
        if semantic_judge
        else None
    )
    recorded = TrajectoryStep(
        input_state=case.input_prompt,
        model_response="recorded evaluation",
        tool_calls=case.observed_tool_calls,
        tool_responses=case.observed_tool_responses,
        token_count=case.observed_token_count,
        cost_usd=case.observed_cost_usd,
        latency_ms=case.observed_latency_ms,
    )
    result = asyncio.run(
        EvaluationRunner(
            lambda _prompt: recorded,
            semantic_judge=judge,
            semantic_threshold=semantic_threshold,
        ).run_case(case)
    )
    console.print(
        Panel(
            f"Status: {'PASS' if result.passed else 'FAIL'}\n"
            f"Deterministic accuracy: {result.deterministic_accuracy_score:.3f}\n"
            f"Tokens: {result.details.get('token_count', 0)} | Cost: ${result.total_cost_usd:.6f}\n"
            f"Latency: {result.details.get('latency_ms', 0):.1f} ms",
            title=f"Case: {case.id}",
            border_style="green" if result.passed else "red",
        )
    )
    if result.schema_errors:
        console.print(Panel("\n".join(result.schema_errors), title="Evaluation errors", border_style="red"))
    if result.details.get("budget_violations"):
        console.print(Panel("\n".join(result.details["budget_violations"]), title="Budget violations", border_style="yellow"))
    rationale = result.details.get("semantic_rationale") or case.observed_semantic_rationale
    score = result.semantic_intent_score
    if score is None:
        score = case.observed_semantic_score
    if rationale:
        label = f"LLM judge rationale (score {score:.3f})" if score is not None else "LLM judge rationale"
        console.print(Panel(rationale, title=label, border_style="cyan"))
    elif not semantic_judge:
        console.print("[dim]No saved judge rationale. Use --semantic-judge to run the configured judge.[/dim]")

    models = build_argument_models(case)
    calls = case.observed_tool_calls
    responses = case.observed_tool_responses
    position = 0
    while True:
        if calls:
            call = calls[position]
            console.print(
                Panel(
                    f"Tool: {call.name}\nTurn: {position + 1} / {len(calls)}\n"
                    f"Arguments:\n{json.dumps(call.kwargs, indent=2, ensure_ascii=False, default=str)}",
                    title=f"Trajectory step {position + 1}",
                    border_style="blue",
                )
            )
            model = models.get(call.name)
            if model is None:
                console.print(Panel("No Pydantic argument schema configured.", title="Schema validation"))
            else:
                try:
                    model.model_validate(call.kwargs)
                except ValidationError as exc:
                    error_text = json.dumps(exc.errors(include_url=False), indent=2, ensure_ascii=False, default=str)
                    console.print(
                        Panel(Syntax(error_text, "json"), title="Pydantic schema: INVALID", border_style="red")
                    )
                else:
                    console.print(Panel("Arguments satisfy the configured Pydantic schema.", title="Pydantic schema: valid", border_style="green"))
            if position < len(responses):
                response = responses[position]
                if isinstance(response, str):
                    try:
                        parsed = json.loads(response)
                    except json.JSONDecodeError:
                        console.print(Panel(Text(response), title="Recorded tool / mock API response"))
                    else:
                        console.print(
                            Panel(Syntax(json.dumps(parsed, indent=2, ensure_ascii=False), "json"), title="Recorded tool / mock API response")
                        )
                else:
                    console.print(
                        Panel(
                            Syntax(json.dumps(response, indent=2, ensure_ascii=False, default=str), "json"),
                            title="Recorded tool / mock API response",
                        )
                    )
            else:
                console.print(Panel("UNKNOWN — this tool response was not recorded.", title="Recorded tool / mock API response"))
        else:
            console.print(Panel("No recorded tool calls for this case.", title="Trajectory"))

        try:
            action = console.input("n (next) / p (previous) / q (quit) > ").strip().lower()
        except EOFError:
            break
        if action in {"q", "quit", "exit"}:
            break
        if action in {"n", "next"}:
            position = min(position + 1, max(len(calls) - 1, 0))
        elif action in {"p", "prev", "previous", "back"}:
            position = max(position - 1, 0)
        else:
            console.print("[yellow]Enter n, p, or q.[/yellow]")


if __name__ == "__main__":
    app()
