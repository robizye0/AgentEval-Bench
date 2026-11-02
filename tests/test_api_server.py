import asyncio

import httpx

from agenteval.mock.api_server import create_app
from agenteval.mock.tool_registry import ToolRegistry


def request(method: str, path: str, app, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_mock_api_lists_and_executes_registered_tool():
    registry = ToolRegistry()
    registry.register("lookup", lambda key: {"key": key})
    app = create_app(registry)
    assert request("GET", "/health", app).json() == {"status": "ok"}
    assert request("GET", "/tools", app).json() == {"tools": ["lookup"]}
    response = request("POST", "/tools/lookup", app, json={"arguments": {"key": "a"}})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "result": {"key": "a"}}


def test_mock_api_maps_injected_http_500_to_server_error():
    registry = ToolRegistry()
    registry.register("lookup", lambda: "ok")
    response = request(
        "POST",
        "/tools/lookup",
        create_app(registry),
        json={"arguments": {}, "failure": "http_500"},
    )
    assert response.status_code == 500


def test_mock_api_injects_rate_limit_and_malformed_json():
    registry = ToolRegistry()
    registry.register("lookup", lambda: "ok")
    app = create_app(registry)
    limited = request("POST", "/tools/lookup", app, json={"arguments": {}, "failure": "rate_limit"})
    malformed = request(
        "POST", "/tools/lookup", app, json={"arguments": {}, "failure": "malformed_json"}
    )
    assert limited.status_code == 429
    assert malformed.status_code == 200
    try:
        malformed.json()
    except ValueError:
        pass
    else:
        raise AssertionError("malformed_json scenario returned valid JSON")
