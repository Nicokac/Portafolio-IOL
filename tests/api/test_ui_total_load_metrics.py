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
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE=")
    monkeypatch.setenv("IOL_TOKENS_KEY", "QkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkI=")

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

    for name in [
        "api.routers.metrics",
        "api.routers",
        "api.main",
        "services.performance_timer",
    ]:
        sys.modules.pop(name, None)

    timer_module = importlib.import_module("services.performance_timer")
    importlib.import_module("api.routers.metrics")
    app_module = importlib.import_module("api.main")
    return timer_module, app_module.app


def test_metrics_endpoint_exposes_ui_total_load_gauge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timer, app = _prepare_app(tmp_path, monkeypatch, enable_prometheus=True)

    timer.update_ui_total_load_metric(8532)
    timer.update_ui_startup_load_metric(1450)
    timer._shutdown_listener()

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "ui_total_load_ms" in response.text
    assert "ui_total_load_ms 8532.0" in response.text
    assert "ui_startup_load_ms" in response.text
    assert "ui_startup_load_ms 1450.0" in response.text
