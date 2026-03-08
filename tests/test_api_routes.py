from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root_route_exists():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "agentic-rag-engine"
    assert payload["docs"] == "/docs"


def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
