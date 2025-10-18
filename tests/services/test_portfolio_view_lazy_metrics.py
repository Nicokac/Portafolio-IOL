import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from services.portfolio_view import PortfolioViewModelService
import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from services.portfolio_view import PortfolioViewModelService


@pytest.fixture
def controls() -> SimpleNamespace:
    return SimpleNamespace(
        hide_cash=False,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
        refresh_secs=30,
        top_n=5,
        order_by="valor_actual",
        desc=True,
        show_usd=False,
    )


@pytest.fixture
def df_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["AL30", "GGAL"],
            "tipo": ["Bono", "Acción"],
            "valor_actual": [120.0, 80.0],
            "costo": [100.0, 70.0],
            "pl": [20.0, 10.0],
            "pl_d": [5.0, 3.0],
            "pl_pct": [18.0, 12.0],
        }
    )


@pytest.fixture
def telemetry_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    files = (tmp_path / "metrics_14.csv", tmp_path / "metrics_15.csv")
    monkeypatch.setattr(
        "shared.telemetry.DEFAULT_TELEMETRY_FILES",
        files,
    )
    return files


@pytest.fixture
def portfolio_service(monkeypatch: pytest.MonkeyPatch) -> PortfolioViewModelService:
    monkeypatch.setattr(
        PortfolioViewModelService,
        "_schedule_snapshot_persistence",
        lambda self, **_: None,
    )
    return PortfolioViewModelService()


def _fake_apply(counter):
    def _apply(df_pos, controls, cli, psvc, *, dataset_hash=None):  # noqa: ANN001 - signature compatibility
        counter["count"] += 1
        return df_pos.copy()

    return _apply


def _basic_args(controls: SimpleNamespace):
    return {
        "controls": controls,
        "cli": SimpleNamespace(),
        "psvc": SimpleNamespace(),
    }


def test_build_minimal_viewmodel_marks_pending(
    monkeypatch: pytest.MonkeyPatch,
    portfolio_service: PortfolioViewModelService,
    controls: SimpleNamespace,
    df_positions: pd.DataFrame,
) -> None:
    counter = {"count": 0}
    monkeypatch.setattr("services.portfolio_view._apply_filters", _fake_apply(counter))

    snapshot = portfolio_service.build_minimal_viewmodel(
        df_positions,
        **_basic_args(controls),
    )

    assert counter["count"] == 1
    assert "extended_metrics" in snapshot.pending_metrics
    cache = portfolio_service._incremental_cache  # type: ignore[attr-defined]
    assert tuple(cache.get("pending_metrics", ())) == ("extended_metrics",)
    assert snapshot.contribution_metrics.by_symbol.empty
    assert snapshot.historical_total.empty


def test_compute_extended_metrics_completes_pending(
    monkeypatch: pytest.MonkeyPatch,
    portfolio_service: PortfolioViewModelService,
    controls: SimpleNamespace,
    df_positions: pd.DataFrame,
) -> None:
    counter = {"count": 0}
    monkeypatch.setattr("services.portfolio_view._apply_filters", _fake_apply(counter))

    portfolio_service.build_minimal_viewmodel(df_positions, **_basic_args(controls))
    assert counter["count"] == 1

    snapshot = portfolio_service.compute_extended_metrics(
        df_positions,
        **_basic_args(controls),
    )

    assert counter["count"] == 1, "La fase extendida no debe recalcular posiciones"
    assert "extended_metrics" not in snapshot.pending_metrics
    cache = portfolio_service._incremental_cache  # type: ignore[attr-defined]
    assert tuple(cache.get("pending_metrics", ())) == ()
    assert not snapshot.contribution_metrics.by_symbol.empty
    assert not snapshot.historical_total.empty


def test_lazy_metrics_phases_logged(
    monkeypatch: pytest.MonkeyPatch,
    portfolio_service: PortfolioViewModelService,
    controls: SimpleNamespace,
    df_positions: pd.DataFrame,
    telemetry_files: tuple[Path, Path],
) -> None:
    counter = {"count": 0}
    monkeypatch.setattr("services.portfolio_view._apply_filters", _fake_apply(counter))

    portfolio_service.build_minimal_viewmodel(df_positions, **_basic_args(controls))
    portfolio_service.compute_extended_metrics(df_positions, **_basic_args(controls))

    for path in telemetry_files:
        assert path.exists(), f"Missing telemetry file {path}"
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        phases = [row["phase"] for row in rows]
        assert "portfolio_view.apply_basic" in phases
        assert "portfolio_view.apply_extended" in phases
