from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import portfolio_view


def _sample_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["GGAL", "AL30"],
            "valor_actual": [100.0, 200.0],
            "costo": [80.0, 150.0],
        }
    )


def test_version_bump_reuses_dataset_but_recomputes_totals(monkeypatch) -> None:
    df_positions = _sample_positions()
    controls = object()

    monkeypatch.setattr(
        portfolio_view,
        "_apply_filters",
        lambda df_pos, controls_unused, cli, psvc, *, dataset_hash=None, skip_invalidation=False: df_pos.copy(),
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

    totals_versions: list[int] = []

    def _fake_totals(df_view: pd.DataFrame) -> portfolio_view.PortfolioTotals:
        version = portfolio_view.PORTFOLIO_TOTALS_VERSION
        totals_versions.append(version)
        value = 500.0 + float(version)
        return portfolio_view.PortfolioTotals(value, 400.0, value - 400.0, 22.0, 0.0)

    monkeypatch.setattr(portfolio_view, "calculate_totals", _fake_totals)

    service = portfolio_view.PortfolioViewModelService()

    snapshot_one = service.get_portfolio_view(df_positions, controls, cli=None, psvc=None)

    assert totals_versions == [portfolio_view.PORTFOLIO_TOTALS_VERSION]
    assert math.isclose(snapshot_one.totals.total_value, 500.0 + portfolio_view.PORTFOLIO_TOTALS_VERSION)

    base_df = snapshot_one.df_view
    assert isinstance(base_df, pd.DataFrame)

    stats_one = service._last_incremental_stats or {}
    assert "positions_df" in stats_one.get("recomputed_blocks", ())

    new_version = portfolio_view.PORTFOLIO_TOTALS_VERSION + 1
    monkeypatch.setattr(portfolio_view, "PORTFOLIO_TOTALS_VERSION", new_version, raising=False)

    snapshot_two = service.get_portfolio_view(
        df_positions,
        controls,
        cli=None,
        psvc=None,
        skip_invalidation=True,
    )

    assert totals_versions == [new_version - 1, new_version]
    assert snapshot_two.df_view is base_df
    assert math.isclose(snapshot_two.totals.total_value, 500.0 + new_version)
    assert snapshot_two.metadata.get("totals_version") == f"v{new_version}"

    stats_two = service._last_incremental_stats or {}
    assert "positions_df" in stats_two.get("reused_blocks", ())
    assert "totals" in stats_two.get("recomputed_blocks", ())

    assert snapshot_two.dataset_hash != snapshot_one.dataset_hash
