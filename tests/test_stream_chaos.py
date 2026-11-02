import asyncio

import httpx

from agenteval.mock.api_server import create_app
from agenteval.mock.tool_registry import ToolRegistry


def test_stream_endpoint_exposes_partial_body_after_injected_drop():
    registry = ToolRegistry()
    registry.register("chunks", lambda: [b"first", b"second", b"third"])
    app = create_app(registry)

    async def request():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/tools/chunks/stream", json={"arguments": {}, "drop_after_chunks": 2}
            )

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.content == b"firstsecond"
