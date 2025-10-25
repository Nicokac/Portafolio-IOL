import pandas as pd
import pytest

from application.portfolio_service import calc_rows


def _quote_stub(_mercado: str, _simbolo: str) -> dict[str, object]:
    return {"last": 1_367.6, "provider": "cache"}


def _build_positions() -> pd.DataFrame:
    base = {
        "simbolo": "BPOC7",
        "mercado": "bcba",
        "tipo": "Bonos",
        "tipo_iol": "Bonos",
        "tipo_estandar": "Bonos",
        "cantidad": 146.0,
        "costo_unitario": 137_600.0,
        "moneda": "ARS",
        "moneda_origen": "ARS",
        "valorizado": 199_669.6,
        "ultimoPrecio": 1_367.6,
    }
    return pd.DataFrame([base])


def test_truncated_payload_without_fallback_remains_truncated() -> None:
    df_pos = _build_positions()

    result = calc_rows(_quote_stub, df_pos, exclude_syms=[])
    row = result.iloc[0]

    assert row["pricing_source"] != "market_revaluation_fallback"
    assert row["valor_actual"] == pytest.approx(199_669.6)


def test_market_fallback_updates_valuation_and_audit() -> None:
    df_pos = _build_positions()
    df_pos.attrs["market_price_fetcher"] = lambda symbol: (
        137_000.0,
        f"/api/v2/BCBA/Titulos/{symbol}/Cotizacion",
    )

    result = calc_rows(_quote_stub, df_pos, exclude_syms=[])
    row = result.iloc[0]

    expected_total = 137_000.0 * 146.0
    assert row["valor_actual"] == pytest.approx(expected_total)
    assert row["pricing_source"] == "market_revaluation_fallback"

    audit = result.attrs.get("audit", {})
    assert audit.get("override_bopreal_market") is True
    assert audit.get("market_price_source", "").endswith("/Cotizacion")
    assert "timestamp_fallback" in audit


def test_market_fallback_tracks_quotes_hash_changes() -> None:
    df_a = _build_positions()
    df_a.attrs["market_price_fetcher"] = lambda symbol: (
        135_500.0,
        f"/api/v2/BCBA/Titulos/{symbol}/Cotizacion",
    )
    df_a.attrs["quotes_hash"] = "hash-A"

    result_a = calc_rows(_quote_stub, df_a, exclude_syms=[])
    audit_a = result_a.attrs.get("audit", {})
    assert audit_a.get("quotes_hash") == "hash-A"

    df_b = _build_positions()
    df_b.attrs["market_price_fetcher"] = lambda symbol: (
        136_200.0,
        f"/api/v2/BCBA/Titulos/{symbol}/CotizacionDetalle",
    )
    df_b.attrs["quotes_hash"] = "hash-B"

    result_b = calc_rows(_quote_stub, df_b, exclude_syms=[])
    audit_b = result_b.attrs.get("audit", {})
    assert audit_b.get("quotes_hash") == "hash-B"
    assert audit_b.get("market_price_source", "").endswith("/CotizacionDetalle")
