from application.portfolio_service import normalize_positions


def test_normalize_positions_basic():
    payload = {
        "activos": [
            {
                "titulo": {
                    "simbolo": "GGAL",
                    "mercado": "BCBA",
                    "tipo": "Acción",
                },
                "cantidad": 10,
                "costoUnitario": 100,
            }
        ]
    }

    df = normalize_positions(payload)
    assert df.shape == (1, 4)
    row = df.iloc[0]
    assert row["simbolo"] == "GGAL"
    assert row["mercado"] == "bcba"
    assert row["cantidad"] == 10.0
    assert row["costo_unitario"] == 100.0
