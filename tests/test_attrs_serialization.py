from __future__ import annotations

import copy
from datetime import datetime
from threading import RLock
from types import MethodType

import pandas as pd

from application.portfolio_service import calc_rows


class _Helper:
    def __init__(self, value: str) -> None:
        self.value = value

    def method(self) -> str:
        return self.value


def _fake_quote(_market: str, _symbol: str) -> dict[str, object]:
    return {
        "last": 125.5,
        "provider": "iol",
        "moneda_origen": "ARS",
        "fx_aplicado": 1.0,
        "timestamp": datetime(2024, 12, 31, 23, 59, 59),
    }


def test_calc_rows_attrs_are_deepcopyable() -> None:
    helper = _Helper("ok")
    df = pd.DataFrame(
        [
        {
            "simbolo": "GGAL",
            "mercado": "BCBA",
            "cantidad": 10,
            "costo_unitario": 100.0,
            "ultimoPrecio": 120.0,
            "valorizado": None,
        }
    ]
)
    df.attrs["audit"] = {
        "lock": RLock(),
        "method": MethodType(_Helper.method, helper),
        "callable": helper.method,
        "nested": [
            {
                "when": datetime(2025, 1, 1, 12, 0, 0),
                "lock": RLock(),
            }
        ],
    }

    result = calc_rows(_fake_quote, df, [])

    clone = copy.deepcopy(result)
    assert clone.equals(result)

    audit = result.attrs.get("audit")
    assert isinstance(audit, dict)
    assert isinstance(audit["lock"], str)
    assert isinstance(audit["method"], str)
    assert isinstance(audit["callable"], str)
    assert isinstance(audit["nested"], list)
    nested_entry = audit["nested"][0]
    assert isinstance(nested_entry["when"], str)
    assert isinstance(nested_entry["lock"], str)
