import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from controllers.portfolio import filters as filters_mod

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def patched_filters(monkeypatch):
    dummy_stage = SimpleNamespace(duration_ms=12.5, cpu_percent=None, ram_percent=None)

    class DummyTimer:
        def __call__(self, *_args, **_kwargs):
            return self

        def __enter__(self):
            return dummy_stage

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyProfiler:
        def __call__(self, *_args, **_kwargs):
            return self

        def __enter__(self):
            return dummy_stage

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_settings = SimpleNamespace(portfolio_background_jobs=False)
    dummy_st = SimpleNamespace(
        session_state={},
        error=lambda msg: (_ for _ in ()).throw(AssertionError(msg)),
        stop=lambda: (_ for _ in ()).throw(AssertionError("st.stop() no debería invocarse")),
    )

    monkeypatch.setattr(filters_mod, "performance_timer", DummyTimer())
    monkeypatch.setattr(filters_mod, "profile_block", DummyProfiler())
    monkeypatch.setattr(filters_mod, "settings", dummy_settings)
    monkeypatch.setattr(filters_mod, "st", dummy_st)

    yield dummy_stage


def test_apply_filters_filters_positions(patched_filters, monkeypatch):
    df_pos = pd.DataFrame(
        [
            {"simbolo": "AL30", "mercado": "BCBA"},
            {"simbolo": "IOLPORA", "mercado": "BCBA"},
            {"simbolo": "GOOG", "mercado": "NASDAQ"},
        ]
    )
    controls = SimpleNamespace(
        hide_cash=True,
        selected_syms=["AL30", "GOOG"],
        selected_types=["Bono"],
        symbol_query="AL",
    )

    quotes = {
        ("bcba", "AL30"): {"chg_pct": 1.0},
        ("nasdaq", "GOOG"): {"chg_pct": 2.0},
    }
    monkeypatch.setattr(filters_mod, "fetch_quotes_bulk", lambda cli, pairs: quotes)

    class DummyPSvc:
        def calc_rows(self, quote_fn, df, exclude_syms=None):  # noqa: ANN001 - mimic signature
            df = df.copy()
            df["valor_actual"] = 100
            return df

        def classify_asset_cached(self, sym):  # noqa: ANN001 - mimic signature
            return {"AL30": "Bono", "GOOG": "Accion"}.get(str(sym), "Otro")

    df_view = filters_mod.apply_filters(df_pos, controls, cli=None, psvc=DummyPSvc())

    assert list(df_view["simbolo"]) == ["AL30"]
    assert "chg_%" in df_view.columns
