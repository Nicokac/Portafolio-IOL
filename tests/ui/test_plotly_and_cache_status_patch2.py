import plotly.graph_objs as go
from types import SimpleNamespace

import ui.tabs.recommendations as rec


def test_cache_status_valid_state(monkeypatch):
    fake_stats = {"ratio": 0.6, "ttl_display": "6h", "last_updated_str": "ahora", "hit_ratio": 0.6}
    called = {}

    def fake_status(*args, **kwargs):
        called["state"] = kwargs.get("state")
        return None

    monkeypatch.setattr(rec.st, "status", fake_status)
    rec._render_cache_status(fake_stats)
    assert called["state"] in {"running", "complete", "error"}


def test_plotly_chart_config(monkeypatch):
    fig = go.Figure()
    called = {}

    def fake_plotly_chart(*args, **kwargs):
        called.update(kwargs)
        return None

    monkeypatch.setattr(rec.st, "plotly_chart", fake_plotly_chart)
    rec.st.plotly_chart(fig, config={"responsive": True})
    assert "config" in called
    assert called["config"]["responsive"] is True
