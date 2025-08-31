from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class Controls:
    refresh_secs: int = 30
    hide_cash: bool = True
    show_usd: bool = False
    order_by: str = "valor_actual"
    desc: bool = True
    top_n: int = 20
    selected_syms: List[str] = field(default_factory=list)
    selected_types: List[str] = field(default_factory=list)
    symbol_query: str = ""
