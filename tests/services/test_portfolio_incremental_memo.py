import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from services.portfolio_view import PortfolioViewModelService


METRICS_PATH = Path("performance_metrics_12.csv")


@pytest.fixture(autouse=True)
def cleanup_metrics_file() -> None:
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()
    yield
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()


@pytest.fixture
def portfolio_service(monkeypatch) -> PortfolioViewModelService:
    monkeypatch.setattr(
        PortfolioViewModelService,
        "_persist_snapshot",
        lambda self, **kwargs: (None, None),
    )
    return PortfolioViewModelService()


def _make_controls(**overrides):
    payload = {
        "hide_cash": False,
        "selected_syms": [],
        "selected_types": [],
        "symbol_query": "",
        "date_range": ("2024-01-01", "2024-06-30"),
        "fx_rates_hash": "fx:base",
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _positions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["AL30", "GGAL"],
            "mercado": ["BCBA", "BCBA"],
        }
    )


def _fake_apply_factory(counter):
    def _fake_apply(
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash=None,
        skip_invalidation=False,
    ):  # noqa: ANN001 - signature compatibility
        counter["count"] += 1
        base = pd.DataFrame(
            {
                "simbolo": ["AL30", "GGAL"],
                "tipo": ["Bono", "Acción"],
                "valor_actual": [120.0 + counter["count"], 80.0 + counter["count"]],
                "costo": [100.0, 70.0],
                "pl": [20.0 + counter["count"], 10.0 + counter["count"]],
                "pl_d": [5.0, 3.0],
                "pl_pct": [18.0, 12.0],
            }
        )
        base["control_range"] = str(getattr(controls, "date_range", ""))
        return base

    return _fake_apply


def _split_blocks(value: str) -> set[str]:
    return {item for item in (value or "").split(";") if item}


def test_incremental_reuses_positions_block(monkeypatch, portfolio_service) -> None:
    counter = {"count": 0}
    monkeypatch.setattr(
        "services.portfolio_view._apply_filters", _fake_apply_factory(counter)
    )

    service = portfolio_service
    df_pos = _positions_frame()
    controls = _make_controls()

    first = service.get_portfolio_view(
        df_pos,
        controls,
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
    )

    assert counter["count"] == 1
    assert first.totals.total_value > 0.0

    updated_controls = _make_controls(date_range=("2024-01-01", "2024-07-31"))
    refreshed = service.get_portfolio_view(
        df_pos,
        updated_controls,
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
    )

    assert counter["count"] == 1, "apply_filters should not run again"
    assert refreshed.totals.total_value > 0.0

    stats = service._last_incremental_stats  # type: ignore[attr-defined]
    assert stats is not None
    reused = set(stats["reused_blocks"])
    recomputed = set(stats["recomputed_blocks"])
    assert "positions_df" in reused
    assert "returns_df" in recomputed
    assert "totals" in recomputed
    assert stats["memoization_hit_ratio"] == pytest.approx(
        len(reused) / (len(reused) + len(recomputed))
    )

    assert METRICS_PATH.exists()
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    last = rows[-1]
    assert last["filters_changed"] == "true"
    assert _split_blocks(last["reused_blocks"]) == reused
    assert _split_blocks(last["recomputed_blocks"]) == recomputed
    assert float(last["memoization_hit_ratio"]) == pytest.approx(
        stats["memoization_hit_ratio"]
    )


def test_incremental_recomputes_positions_on_base_filter_change(
    monkeypatch, portfolio_service
) -> None:
    counter = {"count": 0}
    monkeypatch.setattr(
        "services.portfolio_view._apply_filters", _fake_apply_factory(counter)
    )

    service = portfolio_service
    df_pos = _positions_frame()

    service.get_portfolio_view(
        df_pos,
        _make_controls(selected_syms=["AL30"]),
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
    )
    assert counter["count"] == 1

    refreshed_controls = _make_controls(selected_syms=["GGAL"])
    service.get_portfolio_view(
        df_pos,
        refreshed_controls,
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
    )

    assert counter["count"] == 2, "positions block should recompute"
    stats = service._last_incremental_stats  # type: ignore[attr-defined]
    assert stats is not None
    assert "positions_df" in set(stats["recomputed_blocks"])
    assert stats["memoization_hit_ratio"] == pytest.approx(0.0)


def test_incremental_metrics_csv_contains_expected_columns(
    monkeypatch, portfolio_service
) -> None:
    counter = {"count": 0}
    monkeypatch.setattr(
        "services.portfolio_view._apply_filters", _fake_apply_factory(counter)
    )

    service = portfolio_service
    df_pos = _positions_frame()

    service.get_portfolio_view(
        df_pos,
        _make_controls(),
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
    )

    assert METRICS_PATH.exists()
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    row = rows[0]
    assert set(row) == {
        "dataset_hash",
        "filters_changed",
        "reused_blocks",
        "recomputed_blocks",
        "total_duration_s",
        "memoization_hit_ratio",
    }
    assert row["dataset_hash"]
    assert row["filters_changed"] in {"true", "false"}
    assert float(row["total_duration_s"]) >= 0.0
    assert 0.0 <= float(row["memoization_hit_ratio"]) <= 1.0
