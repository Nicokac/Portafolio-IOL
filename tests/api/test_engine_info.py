"""Tests for the FastAPI engine router."""

from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from api.main import app


def test_engine_info_endpoint_returns_expected_payload() -> None:
    """The /engine/info endpoint should expose engine metadata."""

    with TestClient(app) as client:
        response = client.get("/engine/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["engine_version"] == "v0.6.3"
    assert "timestamp" in payload

    # Validate ISO 8601 compatibility.
    datetime.fromisoformat(payload["timestamp"])
