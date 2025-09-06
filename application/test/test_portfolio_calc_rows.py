import pandas as pd
import pytest

from application.portfolio_service import PortfolioService


def test_calc_rows_integration():
    payload = {
        "activos": [
            {
                "simbolo": "AL30",
                "mercado": "BCBA",
                "cantidad": 100,
                "costoUnitario": 90,
            }
        ]
    }
    svc = PortfolioService()
    df_pos = svc.normalize_positions(payload)
    assert not df_pos.empty

    def quote_fn(mkt, sym):
        assert mkt == "bcba"
        assert sym == "AL30"
        return {"last": 110.0, "chg_pct": 10.0}

    df = svc.calc_rows(quote_fn, df_pos)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["tipo"] == "Bono"
    assert row["costo"] == pytest.approx(90.0)
    assert row["valor_actual"] == pytest.approx(110.0)
    assert row["pl"] == pytest.approx(20.0)
    assert row["pl_%"] == pytest.approx(22.2222, rel=1e-4)
    assert row["pl_d"] == pytest.approx(10.0)
    assert row["pld_%"] == pytest.approx(10.0)