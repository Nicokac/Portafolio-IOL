import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import portfolio_view


@pytest.fixture
def portfolio_service(monkeypatch):
    portfolio_view.reset_portfolio_cache_metrics()

    def fake_apply_filters(df_pos, controls, cli, psvc):  # noqa: D401 - test stub
        return df_pos.copy()

    def fake_totals(df_view):
        total_val = float(df_view.get("valor_actual", pd.Series(dtype=float)).sum() or 0.0)
        total_cost = float(df_view.get("costo", pd.Series(dtype=float)).sum() or 0.0)
        return portfolio_view.PortfolioTotals(total_val, total_cost, total_val - total_cost, 0.0, 0.0)

    def fake_history(self, totals):  # noqa: ANN001 - mimic bound method
        return pd.DataFrame({"ts": [0.0], "total": [totals.total_value]})

    monkeypatch.setattr(portfolio_view, "_apply_filters", fake_apply_filters)
    monkeypatch.setattr(portfolio_view, "calculate_totals", fake_totals)
    monkeypatch.setattr(
        portfolio_view,
        "_compute_contribution_metrics",
        lambda df: portfolio_view.PortfolioContributionMetrics.empty(),
    )
    monkeypatch.setattr(
        portfolio_view.PortfolioViewModelService,
        "_update_history",
        fake_history,
        raising=False,
    )

    service = portfolio_view.PortfolioViewModelService()
    yield service
    portfolio_view.reset_portfolio_cache_metrics()


def _base_positions():
    return pd.DataFrame(
        {
            "simbolo": ["GGAL"],
            "valor_actual": [100.0],
            "costo": [80.0],
        }
    )


def _controls(**overrides):
    base = {
        "hide_cash": False,
        "selected_syms": [],
        "selected_types": [],
        "symbol_query": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_portfolio_cache_metrics_records_hits_and_misses(portfolio_service):
    df = _base_positions()
    controls = _controls()

    snapshot = portfolio_service.get_portfolio_view(df, controls, cli=None, psvc=None)
    assert snapshot is not None
    snapshot = portfolio_service.get_portfolio_view(df, controls, cli=None, psvc=None)
    assert snapshot is not None

    metrics = portfolio_view.get_portfolio_cache_metrics_snapshot()
    assert metrics.misses == 1
    assert metrics.hits == 1
    assert metrics.portfolio_cache_miss_count == 1
    assert metrics.fingerprint_invalidations.get("dataset_changed") == 1
    assert metrics.cache_miss_reasons.get("dataset_and_filters") == 1
    assert metrics.portfolio_cache_hit_ratio == pytest.approx(50.0)


def test_filters_trigger_invalidation_and_miss(portfolio_service):
    df = _base_positions()
    controls_a = _controls()
    controls_b = _controls(selected_syms=["GGAL"])

    portfolio_service.get_portfolio_view(df, controls_a, cli=None, psvc=None)
    portfolio_service.get_portfolio_view(df, controls_a, cli=None, psvc=None)
    portfolio_service.get_portfolio_view(df, controls_b, cli=None, psvc=None)
    portfolio_service.get_portfolio_view(df, controls_b, cli=None, psvc=None)

    metrics = portfolio_view.get_portfolio_cache_metrics_snapshot()
    assert metrics.portfolio_cache_miss_count == 2
    assert metrics.cache_miss_reasons.get("dataset_and_filters") == 1
    assert metrics.cache_miss_reasons.get("filters_changed") == 1
    assert metrics.fingerprint_invalidations.get("dataset_changed") == 1
    assert metrics.fingerprint_invalidations.get("filters_changed") == 1


def test_unexpected_miss_is_flagged(portfolio_service):
    df = _base_positions()
    controls = _controls()

    portfolio_service.get_portfolio_view(df, controls, cli=None, psvc=None)
    portfolio_service.get_portfolio_view(df, controls, cli=None, psvc=None)

    portfolio_service._snapshot = None  # simulate eviction without fingerprint change

    portfolio_service.get_portfolio_view(df, controls, cli=None, psvc=None)

    metrics = portfolio_view.get_portfolio_cache_metrics_snapshot()
    assert metrics.portfolio_cache_miss_count == 2
    assert metrics.cache_miss_reasons.get("unchanged_fingerprint") == 1
    assert metrics.unnecessary_misses() == 1
