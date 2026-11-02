"""In-memory, ordered telemetry ingestion and SSE event fan-out."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agenteval.core.runner import EvaluationRunner
from agenteval.core.schema import TestCase, ToolCall, TrajectoryStep


class IngestEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    input_prompt: str
    expected_tool_sequence: list[str]
    step_index: int = Field(ge=0)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_responses: list[Any] = Field(default_factory=list)
    model_response: str = ""
    token_count: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    final: bool = False
    required_arguments: dict[str, Any] = Field(default_factory=dict)
    semantic_arguments: list[str] = Field(default_factory=list)
    allowed_additional_tools: list[str] = Field(default_factory=list)
    argument_schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    max_allowed_steps: int = Field(default=10, gt=0)
    max_token_budget: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)

    def test_case(self) -> TestCase:
        return TestCase(
            id=self.case_id,
            input_prompt=self.input_prompt,
            expected_tool_sequence=self.expected_tool_sequence,
            required_arguments=self.required_arguments,
            semantic_arguments=self.semantic_arguments,
            allowed_additional_tools=self.allowed_additional_tools,
            argument_schemas=self.argument_schemas,
            max_allowed_steps=self.max_allowed_steps,
            max_token_budget=self.max_token_budget,
            max_cost_usd=self.max_cost_usd,
        )

    def event_signature(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_unset=True),
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_signature(self) -> str:
        config = self.model_dump(
            exclude={
                "event_id", "step_index", "tool_calls", "tool_responses", "model_response",
                "token_count", "cost_usd", "latency_ms", "final",
            },
            mode="json",
        )
        return json.dumps(config, sort_keys=True, separators=(",", ":"))


class IngestionConflict(ValueError):
    pass


class IngestionStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.processed: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self.subscribers: dict[asyncio.Queue[dict[str, Any]], str | None] = {}
        self.lock = asyncio.Lock()

    async def ingest(self, event: IngestEvent) -> dict[str, Any]:
        async with self.lock:
            event_key = (event.case_id, event.event_id)
            event_signature = event.event_signature()
            if event_key in self.processed:
                processed_signature, response = self.processed[event_key]
                if event_signature != processed_signature:
                    raise IngestionConflict("event_id was already used with a different payload")
                return response
            record = self.records.get(event.case_id)
            if record is None:
                if event.step_index != 0:
                    raise IngestionConflict("first event for a case must have step_index 0")
                record = {
                    "case": event.test_case(),
                    "signature": event.config_signature(),
                    "steps": [],
                    "final": False,
                    "latest": None,
                }
                self.records[event.case_id] = record
            elif record["signature"] != event.config_signature():
                raise IngestionConflict("case configuration changed during an evaluation")
            if record["final"]:
                raise IngestionConflict("evaluation is already final")
            if event.step_index != len(record["steps"]):
                raise IngestionConflict(
                    f"expected step_index {len(record['steps'])}, got {event.step_index}"
                )
            record["steps"].append(event)
            record["final"] = event.final
            calls = [call for item in record["steps"] for call in item.tool_calls]
            responses = [response for item in record["steps"] for response in item.tool_responses]
            trajectory = TrajectoryStep(
                input_state=record["case"].input_prompt,
                model_response="\n".join(item.model_response for item in record["steps"]),
                tool_calls=calls,
                tool_responses=responses,
                token_count=sum(item.token_count for item in record["steps"]),
                cost_usd=sum(item.cost_usd for item in record["steps"]),
                latency_ms=sum(item.latency_ms for item in record["steps"]),
            )
            result = await EvaluationRunner(lambda _prompt: trajectory).run_case(record["case"])
            status = ("passed" if result.passed else "failed") if event.final else "in_progress"
            response = {
                "event_id": event.event_id,
                "case_id": event.case_id,
                "step_index": event.step_index,
                "step_count": len(record["steps"]),
                "final": event.final,
                "status": status,
                "token_count": trajectory.token_count,
                "cost_usd": trajectory.cost_usd,
                "latency_ms": trajectory.latency_ms,
                "result": result.model_dump(mode="json"),
            }
            record["latest"] = response
            self.processed[event_key] = (event_signature, response)
            self._publish(response, event.case_id)
            return response

    def _publish(self, payload: dict[str, Any], case_id: str) -> None:
        for queue, filter_id in tuple(self.subscribers.items()):
            if filter_id is not None and filter_id != case_id:
                continue
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(payload)

    async def subscribe(self, case_id: str | None = None) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        async with self.lock:
            self.subscribers[queue] = case_id
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self.lock:
            self.subscribers.pop(queue, None)

    def dashboard(self, case_id: str | None = None) -> dict[str, Any]:
        records = [
            (key, record)
            for key, record in self.records.items()
            if case_id is None or key == case_id
        ]
        cases = []
        for key, record in records:
            latest = record["latest"] or {}
            cases.append(
                {
                    "case_id": key,
                    "status": latest.get("status", "in_progress"),
                    "step_count": len(record["steps"]),
                    "final": record["final"],
                    "passed": latest.get("result", {}).get("passed"),
                    "result": latest.get("result"),
                    "accuracy_score": latest.get("result", {}).get(
                        "deterministic_accuracy_score",
                        latest.get("result", {}).get("tool_accuracy_score"),
                    ),
                    "token_count": latest.get("token_count", 0),
                    "cost_usd": latest.get("cost_usd", 0),
                    "latency_ms": latest.get("latency_ms", 0),
                }
            )
        return {"case_count": len(cases), "cases": cases}
