import math

import numpy as np
import pandas as pd
import pytest

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
    assert totals.total_cash_combined == 0.0


def test_calculate_totals_empty_dataframe():
    totals = calculate_totals(pd.DataFrame())

    assert totals.total_value == 0.0
    assert totals.total_cost == 0.0
    assert totals.total_pl == 0.0
    assert math.isnan(totals.total_pl_pct)
    assert totals.total_cash_combined == 0.0


def test_calculate_totals_uses_cash_balances_attrs_without_duplication():
    df = pd.DataFrame(
        {
            "valor_actual": [1000.0],
            "costo": [800.0],
            "simbolo": ["IOLPORA"],
        }
    )
    df.attrs["cash_balances"] = {
        "cash_ars": 1000.0,
        "cash_usd": 0.0,
        "usd_rate": 500.0,
    }

    totals = calculate_totals(df)

    assert totals.total_cash == pytest.approx(1000.0)
    assert totals.total_cash_ars == pytest.approx(1000.0)
    assert totals.total_cash_usd == pytest.approx(0.0)
    assert totals.total_cash_combined == pytest.approx(1000.0)
    assert totals.usd_rate == pytest.approx(500.0)


def test_calculate_totals_keeps_cash_rows_when_balances_differ():
    df = pd.DataFrame(
        {
            "valor_actual": [1000.0],
            "costo": [800.0],
            "simbolo": ["PARKING"],
        }
    )
    df.attrs["cash_balances"] = {
        "cash_ars": 500.0,
        "cash_usd": 2.0,
        "cash_usd_ars_equivalent": 1000.0,
    }

    totals = calculate_totals(df)

    assert totals.total_cash == pytest.approx(1000.0)
    assert totals.total_cash_ars == pytest.approx(500.0)
    assert totals.total_cash_usd == pytest.approx(2.0)
    assert totals.total_cash_combined == pytest.approx(2500.0)
    assert totals.usd_rate is None


def test_calculate_totals_handles_usd_cash_equivalent_normalized():
    df = pd.DataFrame(
        {
            "valor_actual": [0.0],
            "costo": [0.0],
            "simbolo": ["DUMMY"],
        }
    )
    df.attrs["cash_balances"] = {
        "cash_ars": 0.0,
        "cash_usd": 401_765.812222,
        "cash_usd_ars_equivalent": 3_615_892.31,
        "usd_rate": 9.0,
    }

    totals = calculate_totals(df)

    assert totals.total_cash == pytest.approx(0.0)
    assert totals.total_cash_ars == pytest.approx(0.0)
    assert totals.total_cash_usd == pytest.approx(401_765.812222)
    assert totals.total_cash_combined == pytest.approx(3_615_892.31)
    assert totals.usd_rate == pytest.approx(9.0)


def test_calculate_totals_avoids_double_count_with_money_market_and_usd():
    df = pd.DataFrame(
        {
            "valor_actual": [3_615_892.31],
            "costo": [0.0],
            "simbolo": ["IOLPORA"],
        }
    )
    df.attrs["cash_balances"] = {
        "cash_ars": 0.0,
        "cash_usd": 401_765.812222,
        "cash_usd_ars_equivalent": 3_615_892.31,
        "usd_rate": 9.0,
    }

    totals = calculate_totals(df)

    assert totals.total_cash == pytest.approx(3_615_892.31)
    assert totals.total_cash_ars == pytest.approx(0.0)
    assert totals.total_cash_usd == pytest.approx(401_765.812222)
    assert totals.total_cash_combined == pytest.approx(3_615_892.31)
    assert totals.usd_rate == pytest.approx(9.0)


def test_detect_currency_uses_overrides():
    assert detect_currency("PRPEDOB", None) == "USD"
    assert detect_currency("alua", "bono") == "ARS"
