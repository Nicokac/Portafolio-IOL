from __future__ import annotations

import contextlib

from services import snapshot_defer
from ui.controllers import portfolio_ui


def test_render_portfolio_ui_marks_ui_idle(monkeypatch):
    calls: list[str] = []
    idle_calls: list[float | None] = []

    def fake_mark_busy() -> None:
        calls.append("busy")

    def fake_mark_idle(*, timestamp: float | None = None) -> None:
        calls.append("idle")
        idle_calls.append(timestamp)

    monkeypatch.setattr(snapshot_defer, "mark_ui_busy", fake_mark_busy)
    monkeypatch.setattr(snapshot_defer, "mark_ui_idle", fake_mark_idle)

    def fake_section(container, cli, fx_rates, **kwargs):
        timings = kwargs.setdefault("timings", {})
        timings["portfolio_ui.total"] = 12.5
        return 30

    def fake_timer(name: str, *, extra: dict[str, object] | None = None):
        return contextlib.nullcontext()

    monkeypatch.setattr(portfolio_ui, "_get_portfolio_section", lambda: fake_section)
    monkeypatch.setattr(portfolio_ui, "_get_performance_timer", lambda: fake_timer)
    monkeypatch.setattr(portfolio_ui, "measure_execution", lambda *a, **k: contextlib.nullcontext())

    st = portfolio_ui.st
    st.session_state.clear()

    refresh_secs = portfolio_ui.render_portfolio_ui(container=None, cli=None, fx_rates=None)

    assert refresh_secs == 30
    assert st.session_state.get("ui_idle") is True
    assert calls == ["busy", "idle"]
    assert len(idle_calls) == 1
