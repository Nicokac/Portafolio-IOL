from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Controls:
    """User interface controls and their default values.

    Attributes:
        refresh_secs: Number of seconds between data refreshes.
        hide_cash: Whether cash positions are hidden from the portfolio view.
        show_usd: Whether values are displayed in U.S. dollars alongside pesos.
        order_by: Column or attribute name used to sort portfolio entries.
        desc: Whether the sorting order is descending.
        top_n: Maximum number of entries to display in summary tables.
        selected_syms: Symbol identifiers currently filtered in the UI.
        selected_types: Instrument types currently filtered in the UI.
        symbol_query: Free-text query applied to symbol filtering.
    """

    refresh_secs: int = 30
    hide_cash: bool = True
    show_usd: bool = False
    order_by: str = "valor_actual"
    desc: bool = True
    top_n: int = 20
    selected_syms: list[str] = field(default_factory=list)
    selected_types: list[str] = field(default_factory=list)
    symbol_query: str = ""

