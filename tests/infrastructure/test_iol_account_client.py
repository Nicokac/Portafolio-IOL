import math

import pytest

from infrastructure.iol.account_client import AccountCashSummary, IOLAccountClient


def _parse(payload):
    client = object.__new__(IOLAccountClient)
    return IOLAccountClient._parse_payload(client, payload)


def test_parse_balances_in_ars_only_returns_expected_cash():
    payload = {
        "disponibleEnPesos": "3615892.31",
        "disponibleEnDolares": "0",
        "cotizacionDolar": "9.0",
    }

    summary = _parse(payload)

    assert math.isclose(summary.cash_ars, 3_615_892.31)
    assert summary.cash_usd == pytest.approx(0.0)
    assert summary.usd_rate == pytest.approx(9.0)
    assert summary.usd_ars_equivalent() == pytest.approx(0.0)


def test_parse_balances_normalizes_usd_already_reported_in_ars():
    payload = {
        "disponibleEnPesos": 0,
        "disponibleEnDolares": 3_615_892.31,
        "cotizacionDolar": 9.0,
        "cuentas": [
            {
                "moneda": "Dolares",
                "disponible": 401_765.812222,
                "cotizacion": 9.0,
            }
        ],
    }

    summary = _parse(payload)
    payload = summary.to_payload()

    assert summary.cash_ars == pytest.approx(0.0)
    assert summary.cash_usd == pytest.approx(401_765.812222)
    assert summary.usd_rate == pytest.approx(9.0)
    assert payload["cash_usd_ars_equivalent"] == pytest.approx(3_615_892.31, rel=1e-6)
    assert payload["cash_ars"] == pytest.approx(0.0)
    assert summary.usd_ars_equivalent() == pytest.approx(3_615_892.31, rel=1e-6)


def test_parse_balances_keeps_real_usd_amounts():
    payload = {
        "disponibleEnPesos": 0,
        "disponibleEnDolares": 1_000.0,
        "cotizacionDolar": 9.0,
        "cuentas": [
            {
                "moneda": "Dolares",
                "disponible": 1_000.0,
                "cotizacion": 9.0,
            }
        ],
    }

    summary = _parse(payload)

    assert summary.cash_ars == pytest.approx(0.0)
    assert summary.cash_usd == pytest.approx(1_000.0)
    assert summary.usd_rate == pytest.approx(9.0)
    assert summary.usd_ars_equivalent() == pytest.approx(9_000.0)


def test_parse_balances_handles_missing_sections():
    summary = _parse({})

    assert isinstance(summary, AccountCashSummary)
    assert summary.cash_ars == pytest.approx(0.0)
    assert summary.cash_usd == pytest.approx(0.0)
    assert summary.usd_rate is None
