"""Optional OpenTelemetry and Langfuse-compatible trace hooks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationTracer:
    """Best-effort span exporter; core evaluations work without telemetry extras."""

    def __init__(self, service_name: str = "agenteval-bench") -> None:
        self.tracer: Any = None
        try:
            from opentelemetry import trace

            self.tracer = trace.get_tracer(service_name)
        except ImportError:
            logger.debug("OpenTelemetry not installed; tracing disabled")

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        if self.tracer is None:
            yield None
            return
        with self.tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(key, value)
            yield span

    def record_result(
        self, case_id: str, passed: bool, accuracy: float, latency_ms: float, cost_usd: float
    ) -> None:
        with self.span(
            "agenteval.case",
            {
                "case.id": case_id,
                "eval.passed": passed,
                "eval.accuracy": accuracy,
                "eval.latency_ms": latency_ms,
                "eval.cost_usd": cost_usd,
            },
        ):
            pass
