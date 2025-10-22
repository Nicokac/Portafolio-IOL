import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from controllers.portfolio import portfolio as portfolio_mod
from domain.models import Controls
from services.notifications import NotificationFlags
from tests.fixtures.common import DummyCtx
from tests.fixtures.streamlit import UIFakeStreamlit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _FavoritesStub:
    def sort_options(self, symbols: list[str]) -> list[str]:
        return list(symbols)

    def default_index(self, options: list[str]) -> int:
        return 0

    def format_symbol(self, symbol: str) -> str:
        return symbol

    def is_favorite(self, symbol: str) -> bool:
        return False


class _NotificationsStub:
    def get_flags(self) -> NotificationFlags:
        return NotificationFlags(False, False, False)


class _ViewServiceStub:
    def __init__(self) -> None:
        self._hash_value = "dataset-1"

    def _hash_dataset(self, df_view: pd.DataFrame) -> str:
        return self._hash_value

    def get_portfolio_view(
        self,
        df_pos: pd.DataFrame,
        controls: Controls,
        cli: Any,
        psvc: Any,
        lazy_metrics: bool = False,
    ) -> SimpleNamespace:
        metrics = SimpleNamespace(ccl_rate=None, all_symbols=["GGAL"])
        return SimpleNamespace(
            positions=df_pos,
            controls=controls,
            metrics=metrics,
            totals=None,
            historical_total=None,
            contributions=None,
            pending_metrics=(),
        )

    def compute_extended_metrics(self, **_: Any) -> None:
        return None


@pytest.fixture()
def fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> UIFakeStreamlit:
    fake = UIFakeStreamlit(radio_sequence=[0, 0])
    monkeypatch.setattr(portfolio_mod, "st", fake)
    monkeypatch.setattr(portfolio_mod, "measure_execution", lambda *_: DummyCtx())
    monkeypatch.setattr(portfolio_mod, "profile_block", lambda *_: DummyCtx())
    return fake


@pytest.fixture()
def view_service(monkeypatch: pytest.MonkeyPatch) -> _ViewServiceStub:
    svc = _ViewServiceStub()
    monkeypatch.setattr(portfolio_mod, "get_portfolio_view_service", lambda factory=None: svc)
    return svc


def _patch_render_helpers(monkeypatch: pytest.MonkeyPatch, calls: dict[str, int]) -> None:
    def _make_updater(key: str):
        def _update(placeholder: Any, **kwargs: Any) -> dict[str, Any]:
            calls[key] = calls.get(key, 0) + 1
            refs = kwargs.get("references")
            if not isinstance(refs, dict):
                refs = {}
            refs["has_positions"] = True
            return refs

        return _update

    monkeypatch.setattr(portfolio_mod, "update_summary_section", _make_updater("summary"))
    monkeypatch.setattr(portfolio_mod, "update_table_data", _make_updater("table"))
    monkeypatch.setattr(portfolio_mod, "update_charts", _make_updater("charts"))


def _prepare_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portfolio_mod, "get_portfolio_service", lambda: object())
    monkeypatch.setattr(portfolio_mod, "get_ta_service", lambda: object())
    monkeypatch.setattr(
        portfolio_mod,
        "get_notifications_service",
        lambda factory=None: _NotificationsStub(),
    )
    monkeypatch.setattr(portfolio_mod, "_get_cached_favorites", lambda: _FavoritesStub())
    monkeypatch.setattr(portfolio_mod, "render_sidebar", lambda *_: Controls())
    monkeypatch.setattr(
        portfolio_mod,
        "build_portfolio_viewmodel",
        lambda snapshot, controls, fx_rates, all_symbols: SimpleNamespace(
            controls=controls,
            metrics=snapshot.metrics,
            positions=snapshot.positions,
            totals=None,
            historical_total=None,
            contributions=None,
            pending_metrics=snapshot.pending_metrics,
            tab_options=["Resumen"],
        ),
    )
    monkeypatch.setattr(
        portfolio_mod,
        "load_portfolio_data",
        lambda cli, psvc: (
            pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [100.0]}),
            ["GGAL"],
            ["CEDEAR"],
        ),
    )
    monkeypatch.setattr(portfolio_mod.snapshot_service, "is_null_backend", lambda: False)
    monkeypatch.setattr(portfolio_mod, "_maybe_reset_visual_cache_state", lambda: False)


def test_incremental_overhead_metrics(
    monkeypatch: pytest.MonkeyPatch,
    fake_streamlit: UIFakeStreamlit,
    view_service: _ViewServiceStub,
) -> None:
    _prepare_environment(monkeypatch)
    render_calls: dict[str, int] = {}
    _patch_render_helpers(monkeypatch, render_calls)

    metrics_log: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        portfolio_mod,
        "log_telemetry",
        lambda files,
        phase,
        elapsed_s=None,
        dataset_hash=None,
        extra=None,
        memo_hit_ratio=None,
        subbatch_avg_s=None,
        ui_total_load_ms=None: metrics_log.append((phase, dict(extra or {}))),
    )

    tab_metrics: list[tuple[str, float]] = []
    monkeypatch.setattr(
        portfolio_mod,
        "_append_tab_metric",
        lambda tab, duration, profile_ms=None, overhead_ms=None: tab_metrics.append((tab, duration)),
    )

    tab_latency: list[tuple[str, float, str]] = []
    monkeypatch.setattr(
        portfolio_mod,
        "record_tab_latency",
        lambda slug, value, status: tab_latency.append((slug, value, status)),
    )

    container = DummyCtx()
    portfolio_mod.render_portfolio_section(container, cli=None, fx_rates=None, timings={})

    assert tab_metrics and tab_metrics[0][1] > 0
    assert tab_latency[0][2] in {"fresh", "hot"}

    portfolio_mod.render_portfolio_section(container, cli=None, fx_rates=None, timings={})

    assert render_calls == {"summary": 1, "table": 1, "charts": 1}
    assert tab_latency[-1][2] == "cache"

    visual_logs = [extra for phase, extra in metrics_log if phase == "portfolio.visual_cache"]
    assert visual_logs, "visual cache telemetry should be recorded"
    assert visual_logs[-1]["incremental_render"] is True
    assert visual_logs[-1]["ui_partial_update_ms"] not in (None, "")
