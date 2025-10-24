import math

from application.portfolio_service import calculate_totals, calc_rows, normalize_positions


def test_portfolio_totals_matches_reference_payload() -> None:
    payload = {
        "_cash_balances": {
            "cash_ars": 6_939_049.15,
            "cash_usd": 6.29,
            "cash_usd_ars_equivalent": 9_796.67,
            "usd_rate": 1_557.0,
        },
        "activos": [
            {
                "simbolo": "AL30",
                "mercado": "bcba",
                "cantidad": 1_200.0,
                "costoUnitario": 6_500.0,
                "valorizado": 8_100_000.0,
                "ultimoPrecio": 6_750.0,
                "variacionDiaria": 0.85,
                "titulo": {"tipo": "Bonos Soberanos"},
            },
            {
                "simbolo": "GD30",
                "mercado": "bcba",
                "cantidad": 600.0,
                "costoUnitario": 8_800.0,
                "valorizado": 5_700_000.0,
                "ultimoPrecio": 9_500.0,
                "variacionDiaria": -0.12,
                "titulo": {"tipo": "Bonos Globales"},
            },
            {
                "simbolo": "BPOC7",
                "mercado": "bcba",
                "cantidad": 146.0,
                "costoUnitario": 142_193.15,
                "valorizado": 200_020.0,
                "ultimoPrecio": 137_000.0,
                "variacionDiaria": -1.1,
                "titulo": {"tipo": "Bonos Provinciales"},
            },
            {
                "simbolo": "S10N5",
                "mercado": "bcba",
                "cantidad": 300.0,
                "costoUnitario": 11_500.0,
                "valorizado": 3_750_000.0,
                "ultimoPrecio": 12_500.0,
                "variacionDiaria": 0.35,
                "titulo": {"tipo": "Bonos del Tesoro"},
            },
            {
                "simbolo": "GGAL",
                "mercado": "bcba",
                "cantidad": 320.0,
                "costoUnitario": 5_800.0,
                "valorizado": 1_944_147.92,
                "ultimoPrecio": 6_075.46225,
                "variacionDiaria": 1.25,
                "titulo": {"tipo": "Acciones"},
            },
        ],
    }

    df_pos = normalize_positions(payload)

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {}

    df_rows = calc_rows(_quote_fn, df_pos, exclude_syms=[])
    totals = calculate_totals(df_rows)

    expected_total = 19_694_167.92
    assert math.isclose(totals.total_value, expected_total, rel_tol=0.005)
    assert math.isclose(totals.total_cash_ars, 6_939_049.15, rel_tol=1e-9)
    assert math.isclose(totals.total_cash_usd, 6.29, rel_tol=1e-9)

    combined_expected = 6_939_049.15 + 9_796.67
    assert math.isclose(totals.total_cash_combined, combined_expected, rel_tol=1e-6)

