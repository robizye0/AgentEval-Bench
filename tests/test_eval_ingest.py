import asyncio

import httpx
from fastapi.testclient import TestClient

from agenteval.mock.api_server import create_app
from agenteval.mock.tool_registry import ToolRegistry


def ingest_event(step_index, tools, *, final=False, case_id="remote-1", expected=None):
    return {
        "case_id": case_id,
        "input_prompt": "Find and summarize sources",
        "expected_tool_sequence": expected or ["search", "summarize"],
        "step_index": step_index,
        "event_id": f"{case_id}-{step_index}",
        "tool_calls": [{"name": name, "kwargs": {}} for name in tools],
        "tool_responses": [f"response for {name}" for name in tools],
        "token_count": 5,
        "cost_usd": 0.001,
        "latency_ms": 12,
        "final": final,
    }


def test_remote_ingest_accumulates_steps_and_dashboard_exposes_final_status():
    async def scenario():
        app = create_app(ToolRegistry())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/eval/ingest", json=ingest_event(0, ["search"]))
            second = await client.post(
                "/eval/ingest", json=ingest_event(1, ["summarize"], final=True)
            )
            dashboard = await client.get("/eval/dashboard")
            snapshot = await client.get("/eval/events?once=true")
            return first, second, dashboard, snapshot

    first, second, dashboard, snapshot = asyncio.run(scenario())
    assert first.status_code == 200
    assert first.json()["status"] == "in_progress"
    assert second.json()["status"] == "passed"
    assert second.json()["result"]["passed"] is True
    assert second.json()["token_count"] == 10
    assert dashboard.json()["cases"][0]["status"] == "passed"
    assert dashboard.json()["cases"][0]["accuracy_score"] == 1.0
    assert dashboard.json()["cases"][0]["result"]["passed"] is True
    assert "text/event-stream" in snapshot.headers["content-type"]
    assert "event: snapshot" in snapshot.text
    assert "remote-1" in snapshot.text


def test_remote_ingest_rejects_skipped_step_and_config_drift():
    async def scenario():
        app = create_app(ToolRegistry())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            skipped = await client.post("/eval/ingest", json=ingest_event(1, ["search"]))
            first = await client.post("/eval/ingest", json=ingest_event(0, ["search"]))
            changed = await client.post(
                "/eval/ingest",
                json=ingest_event(1, ["summarize"], expected=["other"]),
            )
            return skipped, first, changed

    skipped, first, changed = asyncio.run(scenario())
    assert skipped.status_code == 409
    assert first.status_code == 200
    assert changed.status_code == 409


def test_ingest_token_auth_is_enforced_when_configured(monkeypatch):
    async def scenario():
        app = create_app(ToolRegistry())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            denied = await client.post("/eval/ingest", json=ingest_event(0, ["search"]))
            accepted = await client.post(
                "/eval/ingest",
                json=ingest_event(0, ["search"], case_id="auth-case"),
                headers={"Authorization": "Bearer test-secret"},
            )
            return denied, accepted

    monkeypatch.setenv("AGENT_EVAL_INGEST_TOKEN", "test-secret")
    denied, accepted = asyncio.run(scenario())
    assert denied.status_code == 401
    assert accepted.status_code == 200


def test_ingest_runs_pydantic_schema_validation_for_live_arguments():
    async def scenario():
        app = create_app(ToolRegistry())
        payload = ingest_event(0, ["search"], final=True, case_id="bad-args")
        payload["tool_calls"] = [{"name": "search", "kwargs": {"count": "many"}}]
        payload["argument_schemas"] = {
            "search": {
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
            }
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/eval/ingest", json=payload)

    response = asyncio.run(scenario())
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert any("Input should be a valid integer" in error for error in response.json()["result"]["schema_errors"])


def test_event_idempotency_is_scoped_to_the_case():
    async def scenario():
        app = create_app(ToolRegistry())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = ingest_event(0, ["search"], case_id="case-a")
            second = ingest_event(0, ["search"], case_id="case-b")
            first["event_id"] = second["event_id"] = "shared-event-id"
            result_a = await client.post("/eval/ingest", json=first)
            result_b = await client.post("/eval/ingest", json=second)
            return result_a, result_b

    result_a, result_b = asyncio.run(scenario())
    assert result_a.json()["case_id"] == "case-a"
    assert result_b.json()["case_id"] == "case-b"


def test_reused_event_id_with_different_payload_is_a_conflict():
    async def scenario():
        app = create_app(ToolRegistry())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            original = ingest_event(0, ["search"], case_id="retry-case")
            first = await client.post("/eval/ingest", json=original)
            exact_retry = await client.post("/eval/ingest", json=original)
            changed = {**original, "token_count": original["token_count"] + 1}
            conflict = await client.post("/eval/ingest", json=changed)
            return first, exact_retry, conflict

    first, exact_retry, conflict = asyncio.run(scenario())
    assert first.status_code == 200
    assert exact_retry.status_code == 200
    assert exact_retry.json() == first.json()
    assert conflict.status_code == 409


def test_live_sse_subscribers_receive_each_ingested_evaluation():
    client = TestClient(create_app(ToolRegistry()))
    with client.stream("GET", "/eval/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        lines = iter(response.iter_lines())
        assert next(lines) == "event: snapshot"
        next(lines)
        assert next(lines) == ""
        posted = client.post("/eval/ingest", json=ingest_event(0, ["search"], case_id="live-sse"))
        assert posted.status_code == 200
        event_lines = []
        for line in lines:
            event_lines.append(line)
            if line.startswith("data: ") and "live-sse" in line:
                break
        assert "event: evaluation" in event_lines
        assert any("live-sse" in line for line in event_lines)
