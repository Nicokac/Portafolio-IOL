"""UI-focused tests for portfolio chart rendering helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.portfolio_service import calculate_totals
from controllers.portfolio import charts as charts_mod
from ui import tables as tables_mod
from domain.models import Controls


class _Column:
    def __init__(self, owner: "_FakeStreamlit") -> None:
        self._owner = owner

    def __enter__(self) -> "_Column":  # noqa: D401 - plain context manager
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - plain context manager
        return None

    def metric(self, *args, **kwargs):  # noqa: ANN002 - proxy helper
        return self._owner.metric(*args, **kwargs)


class _FakeStreamlit:
    """Minimal stub capturing calls performed by ``render_basic_section``."""

    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.subheaders: list[str] = []
        self.infos: list[str] = []
        self.captions: list[str] = []
        self.plot_calls: list[dict[str, object]] = []
        self.selectboxes: list[dict[str, object]] = []
        self.metrics: list[tuple[str, object, object | None]] = []
        self.text_inputs: list[dict[str, object]] = []

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def markdown(self, text: str, **_: object) -> None:
        self.captions.append(text)

    def selectbox(self, label: str, options, index: int = 0, **kwargs):  # noqa: ANN001 - mimic streamlit signature
        self.selectboxes.append({"label": label, "options": list(options)})
        if not options:
            return None
        return options[index]

    def plotly_chart(self, fig, **kwargs):  # noqa: ANN001 - mimic streamlit signature
        self.plot_calls.append({"fig": fig, "kwargs": kwargs})

    def dataframe(self, data, **_: object) -> None:  # noqa: ANN001 - mimic streamlit signature
        self.table = data

    def metric(self, label: str, value: object, delta: object | None = None) -> None:
        self.metrics.append((label, value, delta))

    def columns(self, spec):  # noqa: ANN001 - mimic streamlit signature
        if isinstance(spec, int):
            return [_Column(self) for _ in range(spec)]
        return [_Column(self) for _ in spec]

    def text_input(self, label: str, value: str = "", **_: object) -> str:
        self.text_inputs.append({"label": label, "value": value})
        return value


class _DummyFavorites:
    def __init__(self, items: list[str] | None = None) -> None:
        self._items = items or []

    def list(self) -> list[str]:
        return list(self._items)

    def sort_options(self, options):  # noqa: ANN001 - small helper
        return list(options)

    def default_index(self, options):  # noqa: ANN001 - small helper
        return 0 if options else 0

    def format_symbol(self, symbol):  # noqa: ANN001 - small helper
        return symbol


@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    fake = _FakeStreamlit()
    monkeypatch.setattr(charts_mod, "st", fake)
    monkeypatch.setattr(tables_mod, "st", fake)
    return fake


@pytest.fixture
def _patch_favorite_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(charts_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_table", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_totals", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_portfolio_exports", lambda *a, **k: None)


def _portfolio_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "simbolo": "GGAL",
                "tipo": "ACCION",
                "valor_actual": 150_000.0,
                "costo": 120_000.0,
                "pl": 25_000.0,
                "pl_d": 1_500.0,
            },
            {
                "simbolo": "AAPL",
                "tipo": "CEDEAR",
                "valor_actual": 80_000.0,
                "costo": 70_000.0,
                "pl": -5_000.0,
                "pl_d": -600.0,
            },
        ]
    )


def test_render_basic_section_plots_all_charts(
    fake_streamlit: _FakeStreamlit,
    _patch_favorite_helpers,
) -> None:
    controls = Controls(top_n=3)
    df = _portfolio_frame()

    totals = calculate_totals(df.assign(costo=[120_000.0, 70_000.0]))

    charts_mod.render_basic_section(
        df,
        controls,
        ccl_rate=850.0,
        totals=totals,
        favorites=_DummyFavorites(["GGAL"]),
    )

    keys = {call["kwargs"].get("key") for call in fake_streamlit.plot_calls}
    assert {"pl_topn", "donut_tipo", "dist_tipo", "pl_diario"}.issubset(keys)
    assert fake_streamlit.selectboxes, "Expected favorites selectbox to be rendered"


def test_render_basic_section_handles_empty_frame(
    fake_streamlit: _FakeStreamlit,
    _patch_favorite_helpers,
) -> None:
    charts_mod.render_basic_section(
        pd.DataFrame(),
        Controls(),
        ccl_rate=None,
        totals=None,
        favorites=_DummyFavorites(),
    )

    assert "No hay datos del portafolio para mostrar." in fake_streamlit.infos
