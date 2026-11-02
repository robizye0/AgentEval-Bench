"""Diagnostic, non-destructive suggestions for reducing verbose agent context."""
from __future__ import annotations

import html
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["prompt_history", "tool_response", "tool_arguments"]
    name: str
    content: str
    token_count: int | None = Field(default=None, ge=0)


class CompactionSuggestion(BaseModel):
    segment: str
    kind: str
    estimated_tokens: int
    estimated_savings_tokens: int
    noise_ratio: float = Field(ge=0, le=1)
    strategy: str
    example: str


class ContextCompactionReport(BaseModel):
    estimated_total_tokens: int
    suggestions: list[CompactionSuggestion] = Field(default_factory=list)
    note: str = "Token counts are estimates; review recommendations against task accuracy before applying."


class ContextCompactorEvaluator:
    """Find likely prompt/payload bloat and recommend exact, reviewable compaction steps."""

    _style_script = re.compile(r"(?is)<(script|style|noscript)\b[^>]*>.*?</\1\s*>")
    _tags = re.compile(r"(?s)<[^>]+>")
    _html_hint = re.compile(r"(?is)<(?:html|body|div|p|style|script|table|article|section)\b")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return math.ceil(len(text) / 4) if text else 0

    def _visible_text(self, content: str) -> tuple[str, float, bool]:
        is_html = bool(self._html_hint.search(content))
        if is_html:
            without_active = self._style_script.sub(" ", content)
            without_tags = self._tags.sub(" ", without_active)
            visible = " ".join(html.unescape(without_tags).split())
        else:
            visible = " ".join(content.split())
        ratio = 0.0 if not content else max(0.0, min(1.0, 1 - len(visible) / len(content)))
        return visible, ratio, is_html

    def analyze(
        self, segments: list[ContextSegment], *, budget_breached: bool = False
    ) -> ContextCompactionReport:
        suggestions: list[CompactionSuggestion] = []
        total_tokens = 0
        metrics = []
        for segment in segments:
            visible, noise_ratio, is_html = self._visible_text(segment.content)
            estimated = (
                segment.token_count
                if segment.token_count is not None
                else self.estimate_tokens(segment.content)
            )
            visible_tokens = self.estimate_tokens(visible)
            savings = max(0, estimated - visible_tokens)
            total_tokens += estimated
            metrics.append((segment, estimated, savings, noise_ratio, is_html, visible_tokens))

        for segment, estimated, savings, noise_ratio, is_html, visible_tokens in metrics:
            should_suggest = noise_ratio >= 0.2 or estimated >= 1000
            if not should_suggest:
                continue
            if is_html and noise_ratio >= 0.2:
                strategy = (
                    "Extract visible text: remove <style>/<script>/<noscript> blocks and markup, "
                    "collapse whitespace, then retain only relevant result excerpts."
                )
                example = (
                    "clean = re.sub(r'(?is)<(script|style|noscript)\\b[^>]*>.*?</\\1>', ' ', raw_html)\n"
                    "clean = html.unescape(re.sub(r'<[^>]+>', ' ', clean))\n"
                    "clean = re.sub(r'\\s+', ' ', clean).strip()[:8000]"
                )
            elif segment.kind == "prompt_history":
                strategy = (
                    "Keep system instructions and the latest 3 turns verbatim; summarize older history "
                    "into user_goal, constraints, decisions, and unresolved_questions."
                )
                example = (
                    "history_summary = summarize(history, fields=['user_goal','constraints',"
                    "'decisions','unresolved_questions'])\ncontext = [system, history_summary, *last_3_turns]"
                )
            elif segment.kind == "tool_arguments":
                strategy = (
                    "Keep schema-required keys, omit repeated defaults/metadata, and cap long free-text "
                    "arguments at 512 characters with a clear truncation marker."
                )
                example = (
                    "compact_args = {k: (v[:512] + '…[truncated]' if isinstance(v, str) and len(v) > 512 else v) "
                    "for k, v in args.items() if k in required_keys or v not in (None, '', [], {})}"
                )
            else:
                strategy = (
                    "Replace the raw tool output with the relevant fields and top matching excerpts; "
                    "cap each excerpt at 1,000 characters and preserve source identifiers."
                )
                example = (
                    "compact = {'summary': summary, 'sources': [\n"
                    "    {'id': s.id, 'title': s.title, 'excerpt': s.text[:1000]} for s in top_sources[:5]\n"
                    "]}"
                )
            suggestions.append(
                CompactionSuggestion(
                    segment=segment.name,
                    kind=segment.kind,
                    estimated_tokens=estimated,
                    estimated_savings_tokens=savings,
                    noise_ratio=noise_ratio,
                    strategy=strategy,
                    example=example,
                )
            )

        if budget_breached and not suggestions and metrics:
            segment, estimated, savings, noise_ratio, _is_html, _visible_tokens = max(
                metrics, key=lambda item: item[1]
            )
            suggestions.append(
                CompactionSuggestion(
                    segment=segment.name,
                    kind=segment.kind,
                    estimated_tokens=estimated,
                    estimated_savings_tokens=savings,
                    noise_ratio=noise_ratio,
                    strategy="Summarize repeated context, preserve required facts, and cap the largest segment before sending it to the model.",
                    example="compact = summarize(segment, preserve=required_facts)[:8000]",
                )
            )
        return ContextCompactionReport(
            estimated_total_tokens=total_tokens,
            suggestions=suggestions,
        )
