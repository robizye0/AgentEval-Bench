from fastapi.testclient import TestClient

from agenteval.mock.api_server import create_app


def test_dashboard_cors_allows_local_vite_frontend(monkeypatch):
    monkeypatch.delenv("AGENT_EVAL_CORS_ORIGINS", raising=False)
    client = TestClient(create_app())
    response = client.get(
        "/eval/dashboard",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_dashboard_cors_accepts_configured_origin_list(monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_CORS_ORIGINS", "https://dashboard.example,https://preview.example")
    client = TestClient(create_app())
    response = client.options(
        "/eval/ingest",
        headers={
            "Origin": "https://dashboard.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://dashboard.example"
    assert "authorization" in response.headers["access-control-allow-headers"].lower()
