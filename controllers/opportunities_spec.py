from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OpportunitySpec:
    """Static specification describing the opportunities dataset."""

    columns: List[str]


OPPORTUNITIES_SPEC = OpportunitySpec(
    columns=[
        "symbol",
        "name",
        "market",
        "instrument_type",
        "currency",
        "last_price",
        "change_pct",
        "volume_24h",
        "turnover_24h",
        "score",
    ]
)

__all__ = ["OpportunitySpec", "OPPORTUNITIES_SPEC"]
