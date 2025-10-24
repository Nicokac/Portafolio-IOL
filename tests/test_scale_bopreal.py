import math
from types import SimpleNamespace

import pandas as pd
import pytest

from application.portfolio_service import calc_rows
from services import portfolio_view


def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
    return {}


def _build_bopreal_row(**overrides) -> pd.DataFrame:
    base = {
        "simbolo": "BPOC7",
        "mercado": "bcri",
        "tipo": "Bonos",
        "tipo_iol": "Bonos",
        "tipo_estandar": "Bonos",
        "cantidad": 146.0,
        "costo_unitario": 100.0,
        "moneda": "ARS",
        "moneda_origen": "ARS",
        "valorizado": 199_669.6,
        "ultimoPrecio": 1_367.6,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_bopreal_forced_revaluation_uses_ultimo():
    df_pos = _build_bopreal_row()

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(result.iloc[0]["ultimo"], 1_367.6, rel_tol=1e-6)
    expected_total = 1_367.6 * 146.0
    assert math.isclose(result.iloc[0]["valor_actual"], expected_total, rel_tol=1e-6)
    assert result.iloc[0]["pricing_source"] == "ultimoPrecio"

    audit = result.attrs.get("audit", {})
    decisions = audit.get("scale_decisions", []) if isinstance(audit, dict) else []
    assert any(
        isinstance(entry, dict)
        and entry.get("simbolo") == "BPOC7"
        and entry.get("reason") == "override_bopreal_ars_forced_revaluation"
        for entry in decisions
    )


def test_bopreal_postmerge_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    service = portfolio_view.PortfolioViewModelService()

    base_row = {
        "simbolo": "BPOC7",
        "mercado": "BCBA",
        "tipo": "Bonos",
        "tipo_iol": "Bonos",
        "tipo_estandar": "Bonos",
        "cantidad": 14_600.0,
        "costo": 21_000_000.0,
        "pl": -1_033_040.0,
        "pl_d": -55_000.0,
        "moneda": "ARS",
        "moneda_origen": "ARS",
        "ultimoPrecio": 1_367.6,
        "valor_actual": 199_669.6,
        "pricing_source": "payload_valorizado",
        "audit": {
            "scale_decisions": ["override_bopreal_ars_forced_revaluation"],
        },
        "fx_aplicado": 1.0,
    }
    df_pos = pd.DataFrame([base_row])

    def _fake_apply_filters(df_pos_unused, *_args, **_kwargs):
        return df_pos.copy()

    monkeypatch.setattr(portfolio_view, "_apply_filters", _fake_apply_filters)

    controls = SimpleNamespace(selected_syms=[], selected_types=[], symbol_query="")

    snapshot = service.get_portfolio_view(df_pos, controls, cli=None, psvc=None)

    df_view = snapshot.df_view
    assert not df_view.empty

    row = df_view[df_view["simbolo"] == "BPOC7"].iloc[0]
    assert 19_800_000 < row["valor_actual"] < 20_100_000
    assert row["pricing_source"] == "override_bopreal_postmerge"

    audit_value = row.get("audit")
    if isinstance(audit_value, dict):
        decisions = audit_value.get("scale_decisions", [])
        assert "override_bopreal_postmerge" in decisions

    assert 19_800_000 < snapshot.totals.total_value < 20_100_000


def test_bopreal_forced_revaluation_rescales_valorizado_when_ultimo_missing():
    df_pos = _build_bopreal_row(ultimoPrecio=float("nan"))

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    expected_total = 199_669.6 * 100.0
    assert math.isclose(result.iloc[0]["valor_actual"], expected_total, rel_tol=1e-6)
    assert math.isclose(result.iloc[0]["ultimo"], expected_total / 146.0, rel_tol=1e-6)
    assert result.iloc[0]["pricing_source"] == "valorizado_rescaled"

    audit = result.attrs.get("audit", {})
    decisions = audit.get("scale_decisions", []) if isinstance(audit, dict) else []
    assert any(
        isinstance(entry, dict)
        and entry.get("simbolo") == "BPOC7"
        and entry.get("reason") == "override_bopreal_ars_forced_revaluation"
        for entry in decisions
    )
