import math
from typing import Any

import numpy as np
import pytest

from application import portfolio_service as portfolio_mod


def _build_payload() -> dict[str, Any]:
    return {
        "activos": [
            {
                "simbolo": "GGAL",
                "mercado": "bcba",
                "cantidad": 2,
                "costoUnitario": 100.0,
                "titulo": {"tipo": "Accion", "descripcion": "Accion local"},
            }
        ]
    }


def test_calc_rows_emits_structured_log(caplog: Any) -> None:
    df_pos = portfolio_mod.normalize_positions(_build_payload())
    df_pos["ratioCEDEAR"] = [10.0]

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, Any]:
        return {
            "last": 150.0,
            "provider": "iol",
        }

    caplog.set_level("INFO", logger=portfolio_mod.logger.name)
    portfolio_mod.calc_rows(_quote_fn, df_pos, exclude_syms=[])

    record = next(
        (entry for entry in caplog.records if getattr(entry, "event", "") == "portfolio_valuation_sources"),
        None,
    )
    assert record is not None, f"No structured log found: {caplog.records}"
    valuations = getattr(record, "valuations", [])
    assert valuations, "Expected valuation payload in structured log"
    entry = valuations[0]
    assert entry["simbolo"] == "GGAL"
    assert entry["proveedor_utilizado"] == "iol"
    assert entry["ratioCEDEAR"] == pytest.approx(10.0)
    assert entry["fx_aplicado"] is None


def test_safe_mode_blocks_external_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    df_pos = portfolio_mod.normalize_positions(_build_payload())

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, Any]:
        return {
            "last": 200.0,
            "provider": "external",
            "fx_aplicado": 350.0,
        }

    monkeypatch.setattr(portfolio_mod.settings, "SAFE_VALUATION_MODE", False)
    baseline = portfolio_mod.calc_rows(_quote_fn, df_pos, exclude_syms=[])
    assert math.isclose(float(baseline.loc[baseline.index[0], "valor_actual"]), 400.0)

    monkeypatch.setattr(portfolio_mod.settings, "SAFE_VALUATION_MODE", True)
    safe_mode_df = portfolio_mod.calc_rows(_quote_fn, df_pos, exclude_syms=[])
    assert np.isnan(float(safe_mode_df.loc[safe_mode_df.index[0], "valor_actual"]))
