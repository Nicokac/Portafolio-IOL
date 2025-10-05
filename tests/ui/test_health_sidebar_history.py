import importlib
from typing import Any

import pandas as pd
import pytest


def _render(module, metrics: dict[str, Any]) -> None:
    module.st.session_state["health_metrics"] = metrics
    module.render_health_sidebar()


@pytest.fixture
def health_sidebar(streamlit_stub, monkeypatch: pytest.MonkeyPatch):
    import ui.health_sidebar as health_sidebar_module

    module = importlib.reload(health_sidebar_module)
    monkeypatch.setattr(
        module,
        "get_health_metrics",
        lambda: module.st.session_state.get("health_metrics", {}),
    )
    return module


def test_recent_stats_render_charts_and_badge(health_sidebar, streamlit_stub) -> None:
    metrics = {
        "environment_snapshot": {
            "python_version": "3.11.6",
            "streamlit_version": "1.32.0",
            "runtime": "Local",
            "ts": 1700000000.0,
        },
        "authentication": {},
        "iol_refresh": {},
        "yfinance": {},
        "snapshot": {},
        "fx_api": {
            "stats": {"latency": {"samples": [110.0, 95.0, 102.0]}},
        },
        "fx_cache": {
            "stats": {"age": {"samples": [1.0, 1.5, 0.5]}},
        },
        "portfolio": {
            "stats": {"latency": {"samples": [200.0, 180.0, 220.0]}},
        },
        "quotes": {
            "stats": {"latency": {"samples": [320.0, 300.0]}},
        },
        "session_monitoring": {
            "http_errors": {
                "count": 2,
                "last": {
                    "status_code": 502,
                    "method": "get",
                    "url": "/api/portfolio",
                    "detail": "Bad gateway",
                    "ts": 1699999999.0,
                },
            }
        },
    }

    _render(health_sidebar, metrics)

    assert any(
        "🐍" in markdown and "Streamlit" in markdown
        for markdown in streamlit_stub.sidebar.markdowns
    ), "Expected environment badge in sidebar"

    line_charts = [
        element for element in streamlit_stub.sidebar.elements if element["type"] == "line_chart"
    ]
    assert line_charts, "Expected a line chart with latency samples"
    assert isinstance(line_charts[0]["data"], pd.DataFrame)

    area_charts = [
        element for element in streamlit_stub.sidebar.elements if element["type"] == "area_chart"
    ]
    assert area_charts, "Expected an area chart with cache age samples"
    assert isinstance(area_charts[0]["data"], pd.DataFrame)

    assert any(
        "Último error HTTP" in markdown for markdown in streamlit_stub.sidebar.markdowns
    ), "Expected last HTTP error note"


def test_recent_stats_section_handles_missing_samples(health_sidebar, streamlit_stub) -> None:
    metrics: dict[str, Any] = {
        "authentication": {},
        "iol_refresh": {},
        "yfinance": {},
        "snapshot": {},
        "session_monitoring": {},
    }

    _render(health_sidebar, metrics)

    recent_section = [
        markdown
        for markdown in streamlit_stub.sidebar.markdowns
        if "Estadísticas recientes" in markdown
    ]
    assert recent_section, "Section header should be rendered"

    assert "_Sin estadísticas recientes._" in streamlit_stub.sidebar.markdowns
    assert not [
        element
        for element in streamlit_stub.sidebar.elements
        if element["type"] in {"line_chart", "area_chart"}
    ]

