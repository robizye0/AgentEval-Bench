"""FastAPI mock tool execution service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agenteval.mock.tool_registry import ToolRegistry


class CallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    delay_seconds: float = Field(default=0, ge=0, le=60)
    failure: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)


def create_app(registry: ToolRegistry | None = None) -> FastAPI:
    tools = registry or ToolRegistry()
    app = FastAPI(title="AgentEval-Bench Mock Tool API", version="0.1.0")
    app.state.tool_registry = tools

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
            )
            return {"ok": True, "result": result}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="mock tool timed out") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"mock tool failure: {exc}") from exc

    return app


app = create_app()
