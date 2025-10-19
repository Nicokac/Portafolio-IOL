import pandas as pd
import pytest
from types import SimpleNamespace

from services import portfolio_view
from services.portfolio_view import (
    PortfolioTotals,
    PortfolioViewModelService,
    reset_portfolio_cache_metrics,
)


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    reset_portfolio_cache_metrics()
    yield
    reset_portfolio_cache_metrics()


@pytest.fixture
def portfolio_service(monkeypatch):
    counter = {"apply_calls": 0}

    def _fake_apply_filters(
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash=None,
        skip_invalidation=False,
    ):  # noqa: ANN001
        counter["apply_calls"] += 1
        df = df_pos.copy()
        df["valor_actual"] = 100.0
        df["costo"] = 80.0
        df["pl"] = df["valor_actual"] - df["costo"]
        df["pl_d"] = df["pl"]
        df["pl_pct"] = 12.0
        df["tipo"] = "Acción"
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
    monkeypatch.setattr(
        PortfolioViewModelService,
        "_update_history",
        lambda self, totals: pd.DataFrame(
            {
                "timestamp": [0.0],
                "total_value": [totals.total_value],
                "total_cost": [totals.total_cost],
                "total_pl": [totals.total_pl],
            }
        ),
        raising=False,
    )

    service = PortfolioViewModelService()
    yield service, counter


def _positions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["GGAL", "AL30"],
            "mercado": ["BCBA", "BCBA"],
            "valor_actual": [100.0, 90.0],
            "costo": [80.0, 85.0],
        }
    )


def _controls(**overrides) -> SimpleNamespace:
    payload = {
        "hide_cash": False,
        "selected_syms": [],
        "selected_types": [],
        "symbol_query": "",
        "refresh_secs": 10,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_refresh_secs_does_not_invalidate_apply_basic(portfolio_service):
    service, counter = portfolio_service
    df = _positions_frame()
    controls_a = _controls(refresh_secs=15)
    controls_b = _controls(refresh_secs=5)

    snapshot_a = service.build_minimal_viewmodel(
        df,
        controls_a,
        cli=None,
        psvc=None,
        dataset_hash="hash-one",
    )
    snapshot_b = service.build_minimal_viewmodel(
        df,
        controls_b,
        cli=None,
        psvc=None,
        dataset_hash="hash-one",
    )

    assert counter["apply_calls"] == 1
    assert snapshot_b.totals.total_value == pytest.approx(
        snapshot_a.totals.total_value
    )


def test_dataset_cache_reuses_previous_hash(portfolio_service):
    service, counter = portfolio_service
    df = _positions_frame()
    controls = _controls()

    service.get_portfolio_view(
        df,
        controls,
        cli=None,
        psvc=None,
        dataset_hash="dataset-a",
    )
    service.get_portfolio_view(
        df,
        controls,
        cli=None,
        psvc=None,
        dataset_hash="dataset-b",
    )

    assert counter["apply_calls"] == 2

    service.get_portfolio_view(
        df,
        controls,
        cli=None,
        psvc=None,
        dataset_hash="dataset-a",
    )

    assert counter["apply_calls"] == 2
    metrics = portfolio_view.get_portfolio_cache_metrics_snapshot()
    assert metrics.hits >= 1
