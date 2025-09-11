import pandas as pd
import pytest

from application.portfolio_service import calc_rows
from ui.charts import plot_pl_daily_topn


def test_plot_pl_daily_topn_with_missing_chg_pct():
    df_pos = pd.DataFrame(
        [
            {
                "simbolo": "TEST",
                "mercado": "bcba",
                "cantidad": 10,
                "costo_unitario": 100,
            }
        ]
    )

    def fake_quote(mkt, sym):
        assert (mkt, sym) == ("bcba", "TEST")
        return {"ultimo": 110, "cierreAnterior": 100}

    df = calc_rows(fake_quote, df_pos, [])
    row = df.iloc[0]
    assert row["pl_d"] == pytest.approx(100.0)
    assert row["pld_%"] == pytest.approx(10.0)

    fig = plot_pl_daily_topn(df)
    assert fig is not None
    assert fig.data
    assert fig.data[0].customdata[0][0] == pytest.approx(100.0)
    assert fig.data[0].customdata[0][1] == pytest.approx(10.0)

