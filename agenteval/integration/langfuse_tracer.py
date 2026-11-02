"""Optional OpenTelemetry and Langfuse-compatible trace hooks."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationTracer:
    """Best-effort span exporter; core evaluations work without telemetry extras."""

    def __init__(self, service_name: str = "agenteval-bench") -> None:
        self.tracer: Any = None
        self.provider: Any = None
        try:
            from opentelemetry import trace

            endpoint_configured = bool(
                os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            )
            current_provider = trace.get_tracer_provider()
            if endpoint_configured and type(current_provider).__name__ == "ProxyTracerProvider":
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.resources import Resource
                    from opentelemetry.sdk.trace import TracerProvider
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor
                except ImportError:
                    logger.warning(
                        "OTLP endpoint is configured but telemetry extras are missing; "
                        "install agenteval-bench[telemetry] to export spans"
                    )
                else:
                    provider = TracerProvider(
                        resource=Resource.create({"service.name": service_name})
                    )
                    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                    trace.set_tracer_provider(provider)
                    current_provider = trace.get_tracer_provider()
            self.provider = current_provider
            self.tracer = trace.get_tracer(service_name)
            if not endpoint_configured and type(current_provider).__name__ == "ProxyTracerProvider":
                logger.debug("Tracing is a no-op until an OpenTelemetry SDK/exporter is configured")
        except ImportError:
            logger.debug("OpenTelemetry not installed; tracing disabled")

    def flush(self, timeout_millis: int = 5000) -> bool:
        """Flush queued spans before a short-lived CLI process exits."""
        force_flush = getattr(self.provider, "force_flush", None)
        return bool(force_flush(timeout_millis=timeout_millis)) if force_flush else True

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
        self,
        case_id: str,
        passed: bool,
        accuracy: float,
        latency_ms: float,
        cost_usd: float,
        token_count: int = 0,
        semantic_intent_score: float | None = None,
    ) -> None:
        with self.span(
            "agenteval.case",
            {
                "case.id": case_id,
                "eval.passed": passed,
                "eval.accuracy": accuracy,
                "eval.semantic_intent_score": semantic_intent_score,
                "eval.latency_ms": latency_ms,
                "eval.cost_usd": cost_usd,
                "eval.token_count": token_count,
            },
        ):
            pass

    def record_tool_call(self, case_id: str, name: str, latency_ms: float) -> None:
        with self.span(
            "agenteval.tool_call",
            {"case.id": case_id, "tool.name": name, "tool.latency_ms": latency_ms},
        ):
            pass
