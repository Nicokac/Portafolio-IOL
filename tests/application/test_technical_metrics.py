"""Focused tests around lightweight technical analytics helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.ta_service import simple_alerts


def test_simple_alerts_detects_core_signals() -> None:
    """Synthetic indicators should trigger the expected alert messages."""

    df = pd.DataFrame(
        [
            {
                "RSI": 55,
                "SMA_FAST": 10,
                "SMA_SLOW": 11,
                "Close": 102,
                "BB_U": 110,
                "BB_L": 90,
            },
            {
                "RSI": 72,
                "SMA_FAST": 14,
                "SMA_SLOW": 13,
                "Close": 120,
                "BB_U": 118,
                "BB_L": 95,
            },
        ]
    )

    alerts = simple_alerts(df)

    assert any("RSI en sobrecompra" in alert for alert in alerts)
    assert any("Cruce alcista" in alert for alert in alerts)
    assert any("banda superior" in alert for alert in alerts)


def test_simple_alerts_handles_empty_dataframe() -> None:
    """Empty indicator frames should return an empty alert list."""

    assert simple_alerts(pd.DataFrame()) == []
