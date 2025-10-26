from __future__ import annotations

import pandas as pd
import pytest

from services import portfolio_view


def test_postmerge_bopreal_fix_applied_on_viewmodel() -> None:
    qty = 146.0
    cost = 142_193.15 * qty
    truncated_last = 1_377.0
    df_view = pd.DataFrame(
        {
            "simbolo": ["BPOC7"],
            "moneda": ["ARS"],
            "cantidad": [qty],
            "ultimo": [truncated_last],
            "ultimoPrecio": [truncated_last],
            "valor_actual": [truncated_last * qty],
            "valorizado": [truncated_last * qty],
            "costo": [cost],
            "pl": [0.0],
            "pl_%": [0.0],
            "pricing_source": ["manual"],
            "audit": [{}],
        }
    )

    applied = portfolio_view._apply_bopreal_postmerge_patch(df_view)

    assert applied is True

    row = df_view.iloc[0]
    expected_last = truncated_last * portfolio_view.BOPREAL_FORCE_FACTOR
    expected_value = expected_last * qty
    expected_pl = expected_value - cost
    expected_pl_pct = (expected_pl / cost) * 100.0

    assert row["ultimo"] == pytest.approx(expected_last)
    assert row["valor_actual"] == pytest.approx(expected_value)
    assert row["valorizado"] == pytest.approx(expected_value)
    assert row["pl"] == pytest.approx(expected_pl)
    assert row["pl_%"] == pytest.approx(expected_pl_pct)
    assert row["pricing_source"] == "override_bopreal_postmerge"

    audit_entry = row["audit"]
    assert isinstance(audit_entry, dict)
    assert audit_entry.get("bopreal_postmerge_fix") is True
    assert audit_entry.get("bopreal_postmerge_factor") == portfolio_view.BOPREAL_FORCE_FACTOR
