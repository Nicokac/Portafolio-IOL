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


@pytest.mark.parametrize(
    "simbolo, quote_data, tipo, cantidad, costo_unitario",
    [
        (
            "AAPL",
            {"ultimo": 150.0, "cierreAnterior": 145.0},
            "CEDEAR",
            10,
            140.0,
        ),
        (
            "AL30",
            {"ultimoPrecio": 110.0, "cierreAnterior": 100.0},
            "Bono",
            100,
            90.0,
        ),
    ],
)
def test_calc_rows_derives_chg_pct(simbolo, quote_data, tipo, cantidad, costo_unitario):
    """Calcula chg_pct a partir de ultimo y cierreAnterior y valida pl_d."""

    payload = {
        "activos": [
            {
                "simbolo": simbolo,
                "mercado": "BCBA",
                "cantidad": cantidad,
                "costoUnitario": costo_unitario,
            }
        ]
    }
    svc = PortfolioService()
    df_pos = svc.normalize_positions(payload)

    def quote_fn(mkt, sym):
        assert mkt == "bcba"
        assert sym == simbolo
        last = quote_data.get("ultimo", quote_data.get("ultimoPrecio"))
        prev = quote_data["cierreAnterior"]
        chg_pct = (last - prev) / prev * 100.0
        return {**quote_data, "chg_pct": chg_pct}

    df = svc.calc_rows(quote_fn, df_pos)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["tipo"] == tipo

    last = quote_data.get("ultimo", quote_data.get("ultimoPrecio"))
    prev = quote_data["cierreAnterior"]
    pct = (last - prev) / prev * 100.0
    expected_pld = row["valor_actual"] * (pct / 100.0) / (1.0 + pct / 100.0)
    assert row["pl_d"] == pytest.approx(expected_pld)
    assert row["pld_%"] == pytest.approx(pct)

