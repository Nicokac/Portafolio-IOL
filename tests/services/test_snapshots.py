"""Service-level tests for portfolio snapshots and comparisons."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from domain.models import Controls
from services.portfolio_view import PortfolioViewModelService


def _frame(total_value: float) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "simbolo": ["AL30", "GGAL"],
            "mercado": ["BCBA", "BCBA"],
            "tipo": ["Bonos", "Acciones"],
            "valor_actual": [80.0, 120.0],
            "costo": [70.0, 100.0],
            "pl": [10.0, 20.0],
            "pl_d": [2.0, 4.0],
        }
    )
    scale = total_value / base["valor_actual"].sum()
    scaled = base.copy()
    for column in ("valor_actual", "costo", "pl", "pl_d"):
        scaled[column] = base[column] * scale
    return scaled


@pytest.mark.parametrize("totals", [(200.0, 230.0)])
def test_snapshot_history_supports_comparisons(monkeypatch, totals) -> None:
    initial_total, refreshed_total = totals

    frames = [_frame(initial_total), _frame(refreshed_total)]

    def fake_apply(df, controls, cli, psvc):  # noqa: ANN001 - signature mimic
        return frames.pop(0)

    monkeypatch.setattr("services.portfolio_view._apply_filters", fake_apply)

    service = PortfolioViewModelService()
    controls = Controls()
    positions = pd.DataFrame({"simbolo": ["AL30"], "mercado": ["BCBA"]})

    first = service.get_portfolio_view(
        positions, controls, cli=SimpleNamespace(), psvc=SimpleNamespace()
    )

    assert first.historical_total.shape[0] == 1
    assert first.totals.total_value == pytest.approx(initial_total)

    service.invalidate_filters("refresh")

    second = service.get_portfolio_view(
        positions, controls, cli=SimpleNamespace(), psvc=SimpleNamespace()
    )

    assert second.totals.total_value == pytest.approx(refreshed_total)
    assert second.historical_total.shape[0] == 2
    assert second.historical_total["total_value"].iloc[-1] != pytest.approx(
        second.historical_total["total_value"].iloc[0]
    )

    # The original snapshot remains immutable so comparisons can reuse it.
    assert first.historical_total.shape[0] == 1
    assert first.totals.total_value == pytest.approx(initial_total)

    contribution = second.contribution_metrics.by_type
    assert not contribution.empty
    assert set(contribution.columns) >= {"valor_actual", "pl", "pl_pct"}
