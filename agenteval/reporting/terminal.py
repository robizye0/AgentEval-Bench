"""Rich terminal summaries with actionable per-case failure diffs."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from agenteval.core.schema import EvalResult


def render_results(results: Sequence[EvalResult], console: Console | None = None) -> None:
    console = console or Console(stderr=True)
    table = Table(title="AgentEval-Bench results", expand=True)
    table.add_column("Case", style="bold")
    table.add_column("Result")
    table.add_column("Deterministic", justify="right")
    table.add_column("Semantic", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    for result in results:
        details = result.details
        table.add_row(
            str(details.get("case_id", "case")),
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
            f"{result.deterministic_accuracy_score:.2f}",
            "—" if result.semantic_intent_score is None else f"{result.semantic_intent_score:.2f}",
            f"{details.get('latency_ms', 0):.0f} ms",
            str(details.get("token_count", 0)),
            f"${result.total_cost_usd:.4f}",
        )
    console.print(table)
    failures = [result for result in results if not result.passed]
    if failures:
        tree = Tree("[bold red]Failure diffs[/bold red]")
        for result in failures:
            branch = tree.add(f"[red]{result.details.get('case_id', 'case')}[/red]")
            branch.add(f"Expected tools: {result.details.get('expected_tools', [])}")
            branch.add(f"Observed tools: {result.details.get('observed_tools', [])}")
            for issue in (
                result.schema_errors
                + result.details.get("argument_errors", [])
                + result.details.get("budget_violations", [])
            ):
                branch.add(f"[yellow]{issue}[/yellow]")
            if result.details.get("semantic_rationale"):
                branch.add(f"Judge: {result.details['semantic_rationale']}")
        console.print(tree)
