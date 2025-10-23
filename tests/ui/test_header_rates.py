from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ui import header as header_mod


class _Column:
    def __init__(self, owner: "_FakeStreamlit", index: int) -> None:
        self._owner = owner
        self._index = index

    def markdown(self, body: str, **_: Any) -> None:
        self._owner.column_markdowns[self._index].append(body)


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.column_markdowns: list[list[str]] = []
        self.columns_calls: list[int] = []

    def columns(self, count: int) -> list[_Column]:
        self.columns_calls.append(count)
        self.column_markdowns = [[] for _ in range(count)]
        return [_Column(self, idx) for idx in range(count)]


@pytest.fixture(autouse=True)
def _stub_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    palette = SimpleNamespace(
        bg="#FFFFFF",
        highlight_bg="#0044ff11",
        highlight_text="#001133",
        text="#111111",
        accent="#0044ff",
    )
    monkeypatch.setattr(header_mod, "get_active_palette", lambda: palette)


@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    fake = _FakeStreamlit()
    monkeypatch.setattr(header_mod, "st", fake)
    return fake


def test_header_highlights_active_rate(fake_streamlit: _FakeStreamlit) -> None:
    fake_streamlit.session_state[header_mod._PORTFOLIO_FX_LABEL_KEY] = "mep"
    fake_streamlit.session_state[header_mod._PORTFOLIO_FX_VALUE_KEY] = 920.0

    header_mod.render_fx_summary_in_header({"oficial": 870.0, "mep": 920.0})

    assert fake_streamlit.columns_calls == [2]
    official_card, mep_card = fake_streamlit.column_markdowns
    assert any("Referencia" in body for body in official_card)
    assert any("Usado en totales" in body for body in mep_card)


def test_header_highlights_rate_by_value(fake_streamlit: _FakeStreamlit) -> None:
    fake_streamlit.session_state[header_mod._PORTFOLIO_FX_LABEL_KEY] = ""
    fake_streamlit.session_state[header_mod._PORTFOLIO_FX_VALUE_KEY] = 870.0

    header_mod.render_fx_summary_in_header({"oficial": 870.0, "mep": 885.0})

    official_card, mep_card = fake_streamlit.column_markdowns
    assert any("Usado en totales" in body for body in official_card)
    assert any("Referencia" in body for body in mep_card)
