import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from controllers.portfolio import portfolio as portfolio_mod
from domain.models import Controls
from tests.fixtures.common import DummyCtx
from tests.fixtures.streamlit import UIFakeStreamlit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _FavoritesStub:
    def sort_options(self, symbols: list[str]) -> list[str]:
        return list(symbols)

    def default_index(self, options: list[str]) -> int:
        return 0

    def format_symbol(self, symbol: str) -> str:
        return symbol

    def is_favorite(self, symbol: str) -> bool:
        return False


@pytest.fixture(name="fake_streamlit")
def _fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> UIFakeStreamlit:
    fake = UIFakeStreamlit()
    monkeypatch.setattr(portfolio_mod, "st", fake)
    monkeypatch.setattr(portfolio_mod, "measure_execution", lambda *_: DummyCtx())
    return fake


def _patch_render_helpers(monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[int]]) -> None:
    def _make_updater(key: str):
        def _update(placeholder: Any, **kwargs: Any) -> dict[str, Any]:
            calls.setdefault(key, []).append(id(placeholder))
            refs = kwargs.get("references")
            if not isinstance(refs, dict):
                refs = {}
            refs["has_positions"] = True
            return refs

        return _update

    monkeypatch.setattr(portfolio_mod, "update_summary_section", _make_updater("summary"))
    monkeypatch.setattr(portfolio_mod, "update_table_data", _make_updater("table"))
    monkeypatch.setattr(portfolio_mod, "update_charts", _make_updater("charts"))


def _make_viewmodel(df: pd.DataFrame) -> SimpleNamespace:
    controls = Controls()
    metrics = SimpleNamespace(ccl_rate=None)
    return SimpleNamespace(
        controls=controls,
        metrics=metrics,
        positions=df,
        totals=None,
        historical_total=None,
        contributions=None,
        pending_metrics=(),
    )


def test_incremental_render_reuses_placeholders(
    monkeypatch: pytest.MonkeyPatch, fake_streamlit: UIFakeStreamlit
) -> None:
    calls: dict[str, list[int]] = {}
    _patch_render_helpers(monkeypatch, calls)
    monkeypatch.setattr(portfolio_mod, "_get_cached_favorites", lambda: _FavoritesStub())

    df = pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [100.0]})
    viewmodel = _make_viewmodel(df)

    dataset_hash_key = portfolio_mod._DATASET_HASH_STATE_KEY
    fake_streamlit.session_state[dataset_hash_key] = "hash-1"

    tab_cache: dict[str, Any] = {}
    portfolio_mod.render_basic_tab(
        viewmodel,
        favorites=None,
        snapshot=SimpleNamespace(),
        tab_slug="portafolio",
        tab_cache=tab_cache,
        timings={},
        lazy_metrics=False,
    )

    assert len(calls["summary"]) == 1
    assert len(calls["table"]) == 1
    assert len(calls["charts"]) == 1

    render_refs = fake_streamlit.session_state.get(portfolio_mod._RENDER_REFS_STATE_KEY)
    assert isinstance(render_refs, dict)
    assert render_refs["incremental_render"] is False

    portfolio_mod.render_basic_tab(
        viewmodel,
        favorites=None,
        snapshot=SimpleNamespace(),
        tab_slug="portafolio",
        tab_cache=tab_cache,
        timings={},
        lazy_metrics=False,
    )

    assert len(calls["summary"]) == 1
    assert len(calls["table"]) == 1
    assert len(calls["charts"]) == 1

    assert fake_streamlit.session_state.get("portfolio_incremental_render") is True
    assert portfolio_mod._RENDER_REFS_STATE_KEY in fake_streamlit.session_state
