import pandas as pd
from unittest.mock import MagicMock, patch, call

from ui import tables


def test_detect_currency_returns_expected_values():
    assert tables._detect_currency("PRPEDOB", None) == "USD"
    assert tables._detect_currency("ALUA", "BONO") == "ARS"


def test_render_totals_uses_streamlit_metrics():
    df = pd.DataFrame({"valor_actual": [100, 200], "costo": [40, 110]})
    columns = [MagicMock() for _ in range(4)]
    with patch.object(tables.st, "columns", return_value=columns), \
         patch.object(tables, "format_money", lambda v, currency="ARS": f"{v}-{currency}"):
        tables.render_totals(df)
    expected_calls = [
        call("Valorizado", "300.0-ARS"),
        call("Costo", "150.0-ARS"),
        call("P/L", "150.0-ARS", delta="100.00%"),
        call("P/L %", "100.00%"),
    ]
    actual_calls = [c.metric.call_args for c in columns]
    assert actual_calls == expected_calls
