"""FastAPI mock tool execution service."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from collections.abc import AsyncIterable, Iterable
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from agenteval.mock.telemetry import IngestEvent, IngestionConflict, IngestionStore
from agenteval.mock.tool_registry import ToolRegistry


class CallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    delay_seconds: float = Field(default=0, ge=0, le=60)
    failure: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    jitter_ms: int = Field(default=0, ge=0)
    spike_probability: float = Field(default=0, ge=0, le=1)
    latency_spike_ms: float = Field(default=3000, ge=0)
    random_seed: int = 0


class StreamCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    drop_after_chunks: int = Field(ge=0)


def _check_ingest_auth(authorization: str | None) -> None:
    expected = os.getenv("AGENT_EVAL_INGEST_TOKEN")
    if not expected:
        return
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail="valid bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(registry: ToolRegistry | None = None) -> FastAPI:
    tools = registry or ToolRegistry()
    ingestion = IngestionStore()
    app = FastAPI(title="AgentEval-Bench Mock Tool API", version="0.1.0")
    origins = [
        origin.strip()
        for origin in os.getenv(
            "AGENT_EVAL_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        allow_credentials=False,
    )
    app.state.tool_registry = tools
    app.state.eval_ingestion_store = ingestion

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools")
    async def list_tools() -> dict[str, list[str]]:
        return {"tools": tools.names()}

    @app.post("/tools/{name}")
    async def call_tool(name: str, request: CallRequest) -> Any:
        try:
            result = await tools.call(
                name,
                request.arguments,
                delay=request.delay_seconds,
                failure=request.failure,
                timeout=request.timeout_seconds,
                jitter_ms=request.jitter_ms,
                spike_probability=request.spike_probability,
                latency_spike_ms=request.latency_spike_ms,
                random_seed=request.random_seed,
            )
            return {"ok": True, "result": result}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="mock tool timed out") from exc
        except ConnectionResetError as exc:
            raise HTTPException(status_code=503, detail="simulated connection reset") from exc
        except Exception as exc:
            from agenteval.evaluators.resilience import InjectedFailure

            if isinstance(exc, InjectedFailure):
                if exc.scenario == "malformed_json":
                    return Response(
                        content="{malformed", status_code=200, media_type="application/json"
                    )
                statuses = {"http_500": 500, "rate_limit": 429, "timeout": 504}
                raise HTTPException(
                    status_code=statuses.get(exc.scenario, 503), detail=f"mock tool failure: {exc}"
                ) from exc
            raise HTTPException(status_code=503, detail=f"mock tool failure: {exc}") from exc

    @app.post("/eval/ingest")
    async def ingest_trace(
        event: IngestEvent, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _check_ingest_auth(authorization)
        try:
            return await ingestion.ingest(event)
        except IngestionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/eval/dashboard")
    async def eval_dashboard(
        case_id: str | None = None, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _check_ingest_auth(authorization)
        return ingestion.dashboard(case_id)

    @app.get("/eval/events")
    async def eval_events(
        case_id: str | None = None,
        once: bool = False,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        _check_ingest_auth(authorization)

        async def event_stream():
            queue = await ingestion.subscribe(case_id)
            try:
                snapshot = json.dumps(ingestion.dashboard(case_id), ensure_ascii=False)
                yield f"event: snapshot\ndata: {snapshot}\n\n"
                if once:
                    return
                while True:
                    try:
                        update = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    payload = json.dumps(update, ensure_ascii=False)
                    yield f"event: evaluation\nid: {update['event_id']}\ndata: {payload}\n\n"
            finally:
                await ingestion.unsubscribe(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/tools/{name}/stream")
    async def stream_tool(name: str, request: StreamCallRequest) -> StreamingResponse:
        try:
            result = await tools.call(name, request.arguments)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(result, bytes):
            chunks: Iterable[Any] | AsyncIterable[Any] = [result]
        elif isinstance(result, str):
            chunks = [result.encode("utf-8")]
        elif isinstance(result, dict):
            import json

            chunks = [json.dumps(result).encode("utf-8")]
        else:
            chunks = result

        async def response_chunks():
            from agenteval.evaluators.resilience import ResilienceEvaluator

            async for chunk in ResilienceEvaluator.stream_with_chaos(
                chunks, drop_after_chunks=request.drop_after_chunks
            ):
                yield chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8")

        return StreamingResponse(response_chunks(), media_type="application/octet-stream")

    return app


app = create_app()
