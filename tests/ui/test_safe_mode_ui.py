from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
import pytest

from application.portfolio_service import PortfolioTotals, ValuationBreakdown
from ui import summary_metrics as summary_mod


@dataclass
class _Column:
    owner: "_FakeStreamlit"

    def metric(self, label: str, value: Any, delta: Any = None, **kwargs: Any) -> None:
        self.owner.metric(label, value, delta=delta, **kwargs)


class _Container:
    def __init__(self, owner: "_FakeStreamlit") -> None:
        self._owner = owner

    def __enter__(self) -> "_Container":  # noqa: D401 - context proxy
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - context proxy
        return None


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.metrics: list[tuple[str, Any, Any, dict[str, Any]]] = []
        self.markdowns: list[str] = []
        self.captions: list[str] = []

    def radio(
        self,
        label: str,
        options: Sequence[str],
        *,
        key: str | None = None,
        index: int = 0,
        **_: Any,
    ) -> str:
        selection = options[index] if options else ""
        if key and key in self.session_state and self.session_state[key] in options:
            selection = self.session_state[key]
        if key:
            self.session_state[key] = selection
        return selection

    def container(self, **_: Any) -> _Container:
        return _Container(self)

    def columns(self, spec: int | Sequence[Any]) -> list[_Column]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_Column(self) for _ in range(count)]

    def markdown(self, body: str, **_: Any) -> None:
        self.markdowns.append(body)

    def metric(self, label: str, value: Any, delta: Any = None, **kwargs: Any) -> None:
        self.metrics.append((label, value, delta, kwargs))

    def caption(self, text: str) -> None:
        self.captions.append(text)


@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    fake = _FakeStreamlit()
    monkeypatch.setattr(summary_mod, "st", fake)
    return fake


def test_render_summary_metrics_adds_estimation_tooltip(fake_streamlit: _FakeStreamlit) -> None:
    breakdown = ValuationBreakdown(
        confirmed_rows=5,
        confirmed_value=1000.0,
        estimated_rows=2,
        estimated_value=200.0,
        unconverted_rows=0,
        unconverted_value=0.0,
    )
    totals = PortfolioTotals(
        total_value=1200.0,
        total_cost=800.0,
        total_pl=400.0,
        total_pl_pct=50.0,
        total_cash=0.0,
        total_cash_ars=0.0,
        total_cash_usd=0.0,
        total_cash_combined=0.0,
        usd_rate=None,
        valuation_breakdown=breakdown,
    )
    df = pd.DataFrame({"valor_actual": [1200.0]})

    summary_mod.render_summary_metrics(df, totals=totals, ccl_rate=None)

    tooltip_messages = [kwargs.get("help") for _, _, _, kwargs in fake_streamlit.metrics if kwargs.get("help")]
    assert tooltip_messages, "Expected tooltip for estimated valuations"
    assert any(message.startswith("⚠️ Cotizaciones estimadas") for message in tooltip_messages)
