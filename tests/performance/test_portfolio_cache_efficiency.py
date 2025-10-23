from types import SimpleNamespace

import pandas as pd
import pytest

from services import portfolio_view
from services.portfolio_view import PortfolioTotals, PortfolioViewModelService


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    portfolio_view.reset_portfolio_cache_metrics()
    yield
    portfolio_view.reset_portfolio_cache_metrics()


@pytest.fixture
def portfolio_service(monkeypatch):
    def _fake_apply_filters(
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash=None,
        skip_invalidation=False,
    ):  # noqa: ANN001
        df = df_pos.copy()
        df["valor_actual"] = 100.0
        df["costo"] = 82.0
        df["pl"] = df["valor_actual"] - df["costo"]
        df["pl_d"] = df["pl"]
        df["pl_pct"] = 9.5
        df["tipo"] = "Bono"
        return df

    monkeypatch.setattr(portfolio_view, "_apply_filters", _fake_apply_filters)

    def _fake_totals(df_view):
        total_val = float(df_view.get("valor_actual", pd.Series(dtype=float)).sum() or 0.0)
        total_cost = float(df_view.get("costo", pd.Series(dtype=float)).sum() or 0.0)
        return PortfolioTotals(total_val, total_cost, total_val - total_cost, 0.0, 0.0)

    monkeypatch.setattr(portfolio_view, "calculate_totals", _fake_totals)
    monkeypatch.setattr(
        portfolio_view,
        "_compute_contribution_metrics",
        lambda df: portfolio_view.PortfolioContributionMetrics.empty(),
    )

    service = PortfolioViewModelService()
    return service


def _positions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["GGAL", "YPFD"],
            "mercado": ["BCBA", "BCBA"],
            "valor_actual": [100.0, 95.0],
            "costo": [90.0, 85.0],
        }
    )


def _controls() -> SimpleNamespace:
    return SimpleNamespace(
        hide_cash=False,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
        refresh_secs=10,
    )


def test_dataset_cache_hit_ratio_increases(portfolio_service):
    service = portfolio_service
    df = _positions_frame()
    controls = _controls()

    hashes = ["hash-a", "hash-b", "hash-a", "hash-a"]
    for key in hashes:
        service.get_portfolio_view(
            df,
            controls,
            cli=None,
            psvc=None,
            dataset_hash=key,
        )

    metrics = portfolio_view.get_portfolio_cache_metrics_snapshot()
    assert metrics.hits >= 1
    assert metrics.portfolio_cache_hit_ratio > 0.0
