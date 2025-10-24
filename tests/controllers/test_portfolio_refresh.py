"""Regression tests for portfolio summary refresh and dataset invalidation."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from application import portfolio_service as app_portfolio_service
from controllers.portfolio import portfolio
from services import portfolio_view


def _sample_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["GGAL"],
            "valor_actual": [100.0],
            "costo": [80.0],
        }
    )


def _controls() -> SimpleNamespace:
    return SimpleNamespace(
        hide_cash=False,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
    )


def test_totals_version_invalidation_triggers_refresh(monkeypatch) -> None:
    """Changing the totals version must invalidate cached snapshots."""

    monkeypatch.setattr(
        portfolio_view,
        "_apply_filters",
        lambda df_pos, controls, cli, psvc, *, dataset_hash=None, skip_invalidation=False: df_pos.copy(),
    )
    monkeypatch.setattr(
        portfolio_view,
        "_compute_contribution_metrics",
        lambda df: portfolio_view.PortfolioContributionMetrics.empty(),
    )
    monkeypatch.setattr(
        portfolio_view.PortfolioViewModelService,
        "_update_history",
        lambda self, totals: pd.DataFrame({"ts": [0.0], "total": [totals.total_value]}),
        raising=False,
    )

    service = portfolio_view.PortfolioViewModelService()

    call_count = 0

    def fake_totals(df_view: pd.DataFrame) -> portfolio_view.PortfolioTotals:
        nonlocal call_count
        call_count += 1
        return portfolio_view.PortfolioTotals(100.0, 80.0, 20.0, 25.0, 0.0)

    monkeypatch.setattr(portfolio_view, "calculate_totals", fake_totals)

    df = _sample_positions()
    controls = _controls()

    snapshot_one = service.get_portfolio_view(df, controls, cli=None, psvc=None)
    assert call_count == 1
    assert math.isclose(snapshot_one.totals.total_value, 100.0)
    assert snapshot_one.metadata.get("totals_version") == f"v{portfolio_view.PORTFOLIO_TOTALS_VERSION}"

    summary_currency = "ARS"
    summary_hash_one = (
        f"{portfolio._summary_totals_hash(snapshot_one.totals)}|currency:{summary_currency}|"
        f"v{portfolio.PORTFOLIO_TOTALS_VERSION}"
    )

    new_version = portfolio_view.PORTFOLIO_TOTALS_VERSION + 1
    monkeypatch.setattr(portfolio_view, "PORTFOLIO_TOTALS_VERSION", new_version, raising=False)
    monkeypatch.setattr(portfolio, "PORTFOLIO_TOTALS_VERSION", new_version, raising=False)
    monkeypatch.setattr(app_portfolio_service, "PORTFOLIO_TOTALS_VERSION", new_version, raising=False)

    snapshot_two = service.get_portfolio_view(df, controls, cli=None, psvc=None)

    assert call_count == 2, "calculate_totals should run again after version bump"
    assert snapshot_two.dataset_hash != snapshot_one.dataset_hash
    assert math.isclose(snapshot_two.totals.total_value, 100.0)
    assert snapshot_two.metadata.get("totals_version") == f"v{portfolio_view.PORTFOLIO_TOTALS_VERSION}"

    summary_hash_two = (
        f"{portfolio._summary_totals_hash(snapshot_two.totals)}|currency:{summary_currency}|"
        f"v{portfolio.PORTFOLIO_TOTALS_VERSION}"
    )

    assert summary_hash_two != summary_hash_one


def test_skip_invalidation_version_bump_forces_totals_refresh(monkeypatch) -> None:
    """Changing totals version must recompute totals even when skip_invalidation=True."""

    monkeypatch.setattr(
        portfolio_view,
        "_apply_filters",
        lambda df_pos, controls, cli, psvc, *, dataset_hash=None, skip_invalidation=False: df_pos.copy(),
    )
    monkeypatch.setattr(
        portfolio_view,
        "_compute_contribution_metrics",
        lambda df: portfolio_view.PortfolioContributionMetrics.empty(),
    )

    totals_calls: list[float] = []

    def _fake_totals(df_view: pd.DataFrame) -> portfolio_view.PortfolioTotals:
        version = portfolio_view.PORTFOLIO_TOTALS_VERSION
        totals_calls.append(version)
        base_value = 100.0 + float(version)
        return portfolio_view.PortfolioTotals(base_value, 80.0, base_value - 80.0, 25.0, 0.0)

    monkeypatch.setattr(portfolio_view, "calculate_totals", _fake_totals)
    monkeypatch.setattr(
        portfolio_view.PortfolioViewModelService,
        "_update_history",
        lambda self, totals: pd.DataFrame({"ts": [0.0], "total": [totals.total_value]}),
        raising=False,
    )

    service = portfolio_view.PortfolioViewModelService()
    df = _sample_positions()
    controls = _controls()

    initial_version = portfolio_view.PORTFOLIO_TOTALS_VERSION
    snapshot_one = service.get_portfolio_view(df, controls, cli=None, psvc=None)

    assert totals_calls == [initial_version]
    assert math.isclose(snapshot_one.totals.total_value, 100.0 + initial_version)
    assert snapshot_one.metadata.get("totals_version") == f"v{initial_version}"

    new_version = initial_version + 1
    monkeypatch.setattr(portfolio_view, "PORTFOLIO_TOTALS_VERSION", new_version, raising=False)
    monkeypatch.setattr(portfolio, "PORTFOLIO_TOTALS_VERSION", new_version, raising=False)
    monkeypatch.setattr(app_portfolio_service, "PORTFOLIO_TOTALS_VERSION", new_version, raising=False)

    snapshot_two = service.get_portfolio_view(
        df,
        controls,
        cli=None,
        psvc=None,
        skip_invalidation=True,
    )

    assert totals_calls == [initial_version, new_version]
    assert math.isclose(snapshot_two.totals.total_value, 100.0 + new_version)
    assert snapshot_two.metadata.get("totals_version") == f"v{new_version}"

    stats = service._last_incremental_stats or {}
    assert "positions_df" in stats.get("reused_blocks", ())
    assert "totals" in stats.get("recomputed_blocks", ())
