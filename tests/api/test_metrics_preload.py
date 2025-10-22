from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.test_ui_total_load_metrics import _prepare_app


def test_metrics_endpoint_exposes_preload_timings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timer, app = _prepare_app(tmp_path, monkeypatch, enable_prometheus=True)

    timer.update_preload_total_metric(732)
    timer.update_preload_library_metric("pandas", 412)
    timer.update_preload_library_metric("plotly", 15)
    timer.update_preload_library_metric("statsmodels", float("nan"))
    timer._shutdown_listener()

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "preload_total_ms 732.0" in body
    assert "preload_pandas_ms 412.0" in body
    assert "preload_plotly_ms 15.0" in body
    assert "preload_statsmodels_ms NaN" in body
