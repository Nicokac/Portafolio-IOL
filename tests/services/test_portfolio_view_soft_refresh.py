from types import SimpleNamespace

import pandas as pd

from services import portfolio_view


def _base_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["GGAL"],
            "mercado": ["BCBA"],
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
        refresh_secs=10,
    )


def test_skip_invalidation_preserves_snapshot(monkeypatch):
    portfolio_view.reset_portfolio_cache_metrics()

    apply_calls: list[bool] = []

    def fake_apply_filters(
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash=None,
        skip_invalidation=False,
    ):
        apply_calls.append(bool(skip_invalidation))
        df_view = df_pos.copy()
        df_view["valor_actual"] = 100.0
        df_view["costo"] = 80.0
        df_view["pl"] = df_view["valor_actual"] - df_view["costo"]
        df_view["pl_d"] = df_view["pl"]
        df_view["pl_pct"] = 12.5
        df_view["tipo"] = "Acción"
        return df_view

    monkeypatch.setattr(portfolio_view, "_apply_filters", fake_apply_filters)
    monkeypatch.setattr(
        portfolio_view,
        "calculate_totals",
        lambda df: portfolio_view.PortfolioTotals(
            float(df["valor_actual"].sum()),
            float(df["costo"].sum()),
            float(df["pl"].sum()),
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        portfolio_view,
        "_compute_contribution_metrics",
        lambda df: portfolio_view.PortfolioContributionMetrics.empty(),
    )

    events: list[tuple[str, str | None]] = []
    real_log_event = portfolio_view._log_dataset_cache_event

    def capture_event(event: str, dataset_hash: str | None, **extra):
        events.append((event, dataset_hash))
        return real_log_event(event, dataset_hash, **extra)

    monkeypatch.setattr(portfolio_view, "_log_dataset_cache_event", capture_event)

    invalidations: list[str | None] = []
    real_invalidate = portfolio_view.PortfolioViewModelService.invalidate_positions

    def tracking_invalidate(self, dataset_key: str | None = None):
        invalidations.append(dataset_key)
        return real_invalidate(self, dataset_key)

    monkeypatch.setattr(
        portfolio_view.PortfolioViewModelService,
        "invalidate_positions",
        tracking_invalidate,
    )

    service = portfolio_view.PortfolioViewModelService()

    df_pos = _base_positions()
    controls = _controls()

    snapshot_a = service.get_portfolio_view(
        df_pos,
        controls,
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
        dataset_hash="hash-1",
    )
    service._snapshot = None  # type: ignore[attr-defined]
    service._dataset_cache_adapter = None  # type: ignore[attr-defined]
    service._incremental_cache = {}  # type: ignore[attr-defined]
    snapshot_b = service.get_portfolio_view(
        df_pos,
        controls,
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
        dataset_hash="hash-1",
        skip_invalidation=True,
    )

    assert apply_calls == [False, True]
    assert len(invalidations) == 1, "no additional invalidations expected on soft refresh"
    assert ("skip_invalidation_applied", "hash-1") in events
    assert ("skip_invalidation_guarded", "hash-1") in events
    assert snapshot_b.dataset_hash == "hash-1"
    assert getattr(snapshot_b, "soft_refresh_guard", False) is True

    portfolio_view.reset_portfolio_cache_metrics()
