from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from application.portfolio_service import (
    PortfolioService,
    PortfolioTotals,
    validate_portfolio_consistency,
)
from services.portfolio_view import (
    IncrementalComputationResult,
    PortfolioContributionMetrics,
    PortfolioViewModelService,
)
import services.portfolio_view as portfolio_view_mod


def test_portfolio_consistency_no_diff() -> None:
    df_calc = pd.DataFrame(
        [
            {
                "simbolo": "GD30",
                "valor_actual": 1100.0,
                "ppc": 100.0,
                "pl": 100.0,
                "pl_%": 10.0,
                "valorizado": 1100.0,
            }
        ]
    )

    payload = {
        "activos": [
            {
                "simbolo": "GD30",
                "cantidad": 10,
                "costoUnitario": 100.0,
                "valorizado": 1100.0,
                "titulo": {"tipo": "Bono"},
            }
        ]
    }

    result = validate_portfolio_consistency(df_calc, payload)

    assert result["inconsistency_count"] == 0
    assert result["symbols"] == []
    assert result["checked_symbols"] == ["GD30"]


def test_portfolio_consistency_detects_diff(caplog: pytest.LogCaptureFixture) -> None:
    df_calc = pd.DataFrame(
        [
            {
                "simbolo": "GD30",
                "valor_actual": 1155.0,
                "ppc": 100.0,
                "pl": 155.0,
                "pl_%": 15.5,
                "valorizado": 1155.0,
            }
        ]
    )

    payload = {
        "activos": [
            {
                "simbolo": "GD30",
                "cantidad": 10,
                "costoUnitario": 100.0,
                "valorizado": 1100.0,
                "titulo": {"tipo": "Bono"},
            }
        ]
    }

    with caplog.at_level(logging.WARNING):
        result = validate_portfolio_consistency(df_calc, payload)

    assert result["inconsistency_count"] == 1
    assert result["symbols"] == ["GD30"]
    assert any("Inconsistency detected" in record.message for record in caplog.records)


def test_viewmodel_injects_audit_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    service = PortfolioViewModelService()

    df_pos = pd.DataFrame(
        [
            {
                "simbolo": "GD30",
                "mercado": "bcba",
                "cantidad": 10.0,
                "costo_unitario": 100.0,
                "tipo": "Bono",
                "valorizado": 1100.0,
            }
        ]
    )

    df_view = pd.DataFrame(
        [
            {
                "simbolo": "GD30",
                "mercado": "BCBA",
                "tipo": "Bono",
                "cantidad": 10.0,
                "ppc": 100.0,
                "valor_actual": 1100.0,
                "costo": 1000.0,
                "pl": 100.0,
                "pl_%": 10.0,
                "valorizado": 1100.0,
            }
        ]
    )
    df_view.attrs["audit"] = {}

    totals = PortfolioTotals(1100.0, 1000.0, 100.0, 10.0)
    incremental_result = IncrementalComputationResult(
        df_view=df_view,
        totals=totals,
        contribution_metrics=PortfolioContributionMetrics.empty(),
        historical_total=pd.DataFrame(),
        returns_df=pd.DataFrame(),
        apply_elapsed=0.0,
        totals_elapsed=0.0,
        reused_blocks=tuple(),
        recomputed_blocks=("positions_df",),
        duration=0.0,
        extended_computed=True,
    )

    def _fake_compute_incremental_view(**_kwargs):
        return incremental_result

    monkeypatch.setattr(portfolio_view_mod, "compute_incremental_view", _fake_compute_incremental_view)
    monkeypatch.setattr(portfolio_view_mod, "_apply_bopreal_postmerge_patch", lambda _df: False)
    monkeypatch.setattr(portfolio_view_mod, "log_default_telemetry", lambda **_kwargs: None)

    captured: list[tuple[str, dict[str, object]]] = []

    def _capture_metric(name: str, context: dict[str, object] | None = None, **_kwargs: object) -> None:
        captured.append((name, dict(context or {})))

    monkeypatch.setattr(portfolio_view_mod, "log_metric", _capture_metric)

    controls = SimpleNamespace(selected_syms=[], selected_types=[], symbol_query="", order_by="valor_actual", desc=True)

    snapshot = service.apply_dataset_pipeline(
        df_pos=df_pos,
        controls=controls,
        cli=SimpleNamespace(),
        psvc=PortfolioService(),
        dataset_hash=None,
        mode="full",
        skip_invalidation=False,
    )

    audit_section = snapshot.df_view.attrs.get("audit", {})
    checks = audit_section.get("consistency_checks", {})
    assert "GD30" in checks.get("checked_symbols", [])
    assert snapshot.metadata.get("inconsistency_count") == 0
    assert captured and captured[0][0] == "portfolio_consistency"
    assert captured[0][1].get("inconsistencies") == 0
