import pandas as pd

from application.portfolio_service import scale_for


def test_scale_bopreal_ars():
    row = pd.Series(
        {
            "simbolo": "BPOC7",
            "tipo_activo": "Bono",
            "tipo": "Bono",
            "tipo_estandar": "Bono",
            "moneda": "ARS",
            "moneda_origen": "ARS",
            "proveedor_original": "IOL",
            "precio_feed": 1363.1,
            "cantidad": 14_600,
        }
    )
    scale = scale_for(row)
    assert scale == 1.0
    val = row["precio_feed"] * row["cantidad"] * scale
    assert 19_800_000 < val < 20_000_000
