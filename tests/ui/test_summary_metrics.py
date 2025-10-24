from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import pytest

from application.portfolio_service import PortfolioTotals
from ui import summary_metrics as summary_mod


class _Column:
    def __init__(self, owner: "_FakeStreamlit") -> None:
        self._owner = owner

    def metric(self, label: str, value: Any, delta: Any = None, **kwargs: Any) -> None:
        self._owner.metric(label, value, delta=delta, **kwargs)


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
        self.radio_calls: list[dict[str, Any]] = []

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
        self.radio_calls.append({"label": label, "options": list(options), "value": selection})
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


def _make_totals() -> PortfolioTotals:
    return PortfolioTotals(
        total_value=120_000.0,
        total_cost=100_000.0,
        total_pl=20_000.0,
        total_pl_pct=20.0,
        total_cash=5_000.0,
        total_cash_ars=2_000.0,
        total_cash_usd=10.0,
        total_cash_combined=12_000.0,
        usd_rate=1_000.0,
    )


def _metric_by_label(fake: _FakeStreamlit, label: str) -> tuple[Any, dict[str, Any]]:
    for recorded_label, value, _delta, kwargs in fake.metrics:
        if recorded_label == label:
            return value, kwargs
    raise AssertionError(f"Metric {label} no registrada: {fake.metrics}")


def test_render_summary_metrics_switches_currency(fake_streamlit: _FakeStreamlit) -> None:
    df = pd.DataFrame({"valor_actual": [120_000.0]})
    totals = _make_totals()

    summary_mod.render_summary_metrics(df, totals=totals, ccl_rate=None)

    value, _ = _metric_by_label(fake_streamlit, "Valorizado")
    assert value == "$ 120.000,00"

    usd_value, usd_kwargs = _metric_by_label(fake_streamlit, "Cash USD · USD")
    assert usd_value == "US$ 10,00"
    assert "≈ $ 10.000,00 en ARS" in usd_kwargs.get("help", "")
    assert "Moneda base seleccionada: ARS" in fake_streamlit.captions[-1]

    fake_streamlit.metrics.clear()
    fake_streamlit.captions.clear()
    fake_streamlit.session_state[summary_mod.CURRENCY_STATE_KEY] = "USD"

    summary_mod.render_summary_metrics(df, totals=totals, ccl_rate=None)

    value, _ = _metric_by_label(fake_streamlit, "Valorizado")
    assert value == "US$ 120,00"
    cash_total_value, _ = _metric_by_label(fake_streamlit, "Cash total · USD")
    assert cash_total_value == "US$ 17,00"
    usd_value, usd_kwargs = _metric_by_label(fake_streamlit, "Cash USD · USD")
    assert usd_value == "US$ 10,00"
    # En moneda base USD el tooltip conserva la referencia ARS
    assert "≈ $ 10.000,00 en ARS" in usd_kwargs.get("help", "")
    assert "Moneda base seleccionada: USD" in fake_streamlit.captions[-1]


def test_get_active_summary_currency_falls_back_to_ars(fake_streamlit: _FakeStreamlit) -> None:
    fake_streamlit.session_state[summary_mod.CURRENCY_STATE_KEY] = "GBP"
    assert summary_mod.get_active_summary_currency() == "ARS"
