from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _prepare_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_prometheus: bool,
):
    log_path = tmp_path / "api_performance.log"
    structured_path = tmp_path / "api_structured.log"
    monkeypatch.setenv("PERFORMANCE_LOG_PATH", str(log_path))
    monkeypatch.setenv("PERFORMANCE_JSON_LOG_PATH", str(structured_path))

    from shared import settings as shared_settings

    shared_settings.settings.REDIS_URL = None
    shared_settings.settings.ENABLE_PROMETHEUS = enable_prometheus
    shared_settings.enable_prometheus = enable_prometheus
    shared_settings.settings.PERFORMANCE_VERBOSE_TEXT_LOG = False
    shared_settings.performance_verbose_text_log = False
    shared_settings.settings.app_env = "dev"
    shared_settings.app_env = "dev"

    existing_timer = sys.modules.get("services.performance_timer")
    if existing_timer and hasattr(existing_timer, "_shutdown_listener"):
        existing_timer._shutdown_listener()

    for name in ["api.routers.metrics", "api.routers", "api.main", "services.performance_timer"]:
        sys.modules.pop(name, None)

    timer_module = importlib.import_module("services.performance_timer")
    app_module = importlib.import_module("api.main")
    return timer_module, app_module.app


def test_metrics_endpoint_returns_prometheus_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timer, app = _prepare_app(tmp_path, monkeypatch, enable_prometheus=True)

    with timer.performance_timer("api_metrics_test"):
        pass

    timer._shutdown_listener()

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "performance_duration_seconds_count" in response.text
    assert "label=\"api_metrics_test\"" in response.text


def test_metrics_endpoint_respects_disabled_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timer, app = _prepare_app(tmp_path, monkeypatch, enable_prometheus=False)
    assert timer.PROMETHEUS_ENABLED is False
    assert timer.PROMETHEUS_REGISTRY is None
    timer._shutdown_listener()

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 404
