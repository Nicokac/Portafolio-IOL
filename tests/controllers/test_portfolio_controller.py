from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.portfolio import portfolio


class DummyFavorites:
    def sort_options(self, options):
        return list(options)

    def default_index(self, options):
        return 0

    def format_symbol(self, symbol):
        return symbol

    def list(self):
        return []


class DummyUI:
    def __init__(self):
        self._select_calls = 0

    def subheader(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def selectbox(self, _label, options, **_kwargs):
        # return first option deterministically
        return options[0] if options else None

    def columns(self, spec):
        count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)

        class _DummyColumn:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_exc):
                return False

            def number_input(self_inner, _label, **kwargs):
                return kwargs.get("value", 0)

        return [_DummyColumn() for _ in range(count)]

    def number_input(self, _label, **kwargs):
        return kwargs.get("value", 0)

    def expander(self, *_args, **_kwargs):
        return nullcontext()

    def plotly_chart(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def success(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def line_chart(self, *_args, **_kwargs):
        return None

    def metric(self, *_args, **_kwargs):
        return None


def test_render_basic_tab_uses_viewmodel(monkeypatch: pytest.MonkeyPatch) -> None:
    viewmodel = SimpleNamespace(
        positions="positions",
        controls="controls",
        metrics=SimpleNamespace(ccl_rate=1.2),
        totals="totals",
        historical_total="history",
        contributions="contrib",
    )
    snapshot = object()
    favorites = SimpleNamespace()

    calls: dict[str, int] = {"snapshot": 0, "summary": 0, "table": 0, "charts": 0}

    class _Placeholder:
        def empty(self):
            return None

        def container(self):
            return nullcontext()

    def fake_snapshot(vm):
        calls["snapshot"] += 1
        assert vm is viewmodel

    def fake_summary(*args, **kwargs):
        calls["summary"] += 1
        return True

    def fake_table(*args, **kwargs):
        calls["table"] += 1

    def fake_charts(*args, **kwargs):
        calls["charts"] += 1

    monkeypatch.setattr(portfolio, "st", SimpleNamespace(empty=lambda: _Placeholder(), caption=lambda *_: None, spinner=nullcontext))
    monkeypatch.setattr(portfolio, "_ensure_component_store", lambda cache: {})
    monkeypatch.setattr(portfolio, "_ensure_component_entry", lambda store, name: {"placeholder": _Placeholder()})

    monkeypatch.setattr(portfolio, "_render_snapshot_comparison_controls", fake_snapshot)
    monkeypatch.setattr(portfolio, "render_summary_section", fake_summary)
    monkeypatch.setattr(portfolio, "render_table_section", fake_table)
    monkeypatch.setattr(portfolio, "render_charts_section", fake_charts)
    monkeypatch.setattr(portfolio, "_render_updated_caption", lambda *_: None)

    portfolio.render_basic_tab(viewmodel, favorites, snapshot)

    assert calls == {"snapshot": 1, "summary": 1, "table": 1, "charts": 1}


def test_render_notifications_panel_renders_badges(monkeypatch: pytest.MonkeyPatch) -> None:
    favorites = SimpleNamespace()
    notifications = SimpleNamespace(technical_signal=True)
    ui = DummyUI()

    badge_calls: list[tuple] = []
    favorites_calls: list[tuple] = []

    def fake_badge(*args, **kwargs):
        badge_calls.append((args, kwargs))

    def fake_favorites(*args, **kwargs):
        favorites_calls.append((args, kwargs))

    monkeypatch.setattr(portfolio, "render_technical_badge", fake_badge)
    monkeypatch.setattr(portfolio, "render_favorite_badges", fake_favorites)

    portfolio.render_notifications_panel(favorites, notifications, ui=ui)

    assert len(badge_calls) == 1
    assert favorites_calls == [
        (
            (favorites,),
            {"empty_message": "⭐ Aún no marcaste favoritos para seguimiento rápido."},
        )
    ]


def test_render_technical_tab_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    favorites = DummyFavorites()
    notifications = SimpleNamespace(technical_signal=True)
    all_symbols = ["GGAL"]
    viewmodel = SimpleNamespace(metrics=SimpleNamespace(all_symbols=["GGAL"]))

    tasvc = SimpleNamespace(
        fundamentals=lambda _sym: {"pe": 10},
        indicators_for=lambda *args, **kwargs: pd.DataFrame(
            {"close": [1, 2, 3], "equity": [1.0, 1.1, 1.2]}
        ),
        alerts_for=lambda _df: ["Señal alcista"],
        backtest=lambda _df, strategy: pd.DataFrame({"equity": [1.0, 1.2]}),
    )

    monkeypatch.setattr(portfolio, "render_favorite_toggle", lambda *args, **kwargs: None)
    monkeypatch.setattr(portfolio, "render_fundamental_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(portfolio, "plot_technical_analysis_chart", lambda *args, **kwargs: {})
    monkeypatch.setattr(portfolio, "record_tab_latency", lambda *args, **kwargs: None)

    ui = DummyUI()

    portfolio.render_technical_tab(
        tasvc,
        favorites,
        notifications,
        all_symbols,
        viewmodel,
        map_symbol=lambda sym: f"{sym}.AR",
        ui=ui,
        timer=lambda: 1.0,
        render_fundamentals=lambda *_args, **_kwargs: None,
    )
