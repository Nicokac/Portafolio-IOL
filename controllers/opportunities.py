from __future__ import annotations

import pandas as pd

from controllers.opportunities_spec import OPPORTUNITIES_SPEC


def search_opportunities() -> pd.DataFrame:
    """Return a static set of investment opportunities.

    The data is deterministic to keep tests stable and focuses on
    providing every column declared in :data:`OPPORTUNITIES_SPEC`.
    """

    data = [
        {
            "symbol": "AL30",
            "name": "Bono AL30",
            "market": "BCBA",
            "instrument_type": "Bono",
            "currency": "ARS",
            "last_price": 12345.67,
            "change_pct": 1.23,
            "volume_24h": 1_500_000,
            "turnover_24h": 925_000_000.0,
            "score": 78,
        },
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corp.",
            "market": "NASDAQ",
            "instrument_type": "Acción",
            "currency": "USD",
            "last_price": 904.12,
            "change_pct": -0.42,
            "volume_24h": 3_200_000,
            "turnover_24h": 2_893_000_000.0,
            "score": 82,
        },
    ]

    return pd.DataFrame(data, columns=OPPORTUNITIES_SPEC.columns)


__all__ = ["search_opportunities"]
