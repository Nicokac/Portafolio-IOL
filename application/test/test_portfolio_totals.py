import math

import numpy as np
import pandas as pd

from application.portfolio_service import calculate_totals, detect_currency


def test_calculate_totals_basic_dataframe():
    df = pd.DataFrame(
        {
            "valor_actual": [100.0, 200.0, np.nan],
            "costo": [40.0, 110.0, np.nan],
        }
    )

    totals = calculate_totals(df)

    assert totals.total_value == 300.0
    assert totals.total_cost == 150.0
    assert totals.total_pl == 150.0
    assert math.isclose(totals.total_pl_pct, 100.0)


def test_calculate_totals_empty_dataframe():
    totals = calculate_totals(pd.DataFrame())

    assert totals.total_value == 0.0
    assert totals.total_cost == 0.0
    assert totals.total_pl == 0.0
    assert math.isnan(totals.total_pl_pct)


def test_detect_currency_uses_overrides():
    assert detect_currency("PRPEDOB", None) == "USD"
    assert detect_currency("alua", "bono") == "ARS"
