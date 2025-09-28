from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
from types import ModuleType
import sys
import textwrap

import pandas as pd
import streamlit as _streamlit_module
from streamlit.testing.v1 import AppTest
from streamlit.runtime.secrets import Secrets

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def _resolve_streamlit_module() -> ModuleType:
    """Ensure we use the real Streamlit implementation, not a test stub."""
    if getattr(_streamlit_module, "__file__", None) and hasattr(
        _streamlit_module, "button"
    ):
        return _streamlit_module

    for name in list(sys.modules):
        if name == "streamlit" or name.startswith("streamlit."):
            sys.modules.pop(name, None)

    import streamlit as real_streamlit

    return real_streamlit


st = _resolve_streamlit_module()
sys.modules["streamlit"] = st


_ORIGINAL_STREAMLIT = st


def _ensure_real_streamlit_module() -> None:
    global st
    if sys.modules.get("streamlit") is not _ORIGINAL_STREAMLIT:
        sys.modules["streamlit"] = _ORIGINAL_STREAMLIT
    if st is not _ORIGINAL_STREAMLIT:
        st = _ORIGINAL_STREAMLIT
    ui_settings_mod = sys.modules.get("ui.ui_settings")
    if ui_settings_mod is not None:
        setattr(ui_settings_mod, "st", _ORIGINAL_STREAMLIT)


from controllers import opportunities as opportunities_ctrl
from controllers.opportunities_spec import OPPORTUNITIES_SPEC
from shared.version import __version__

_SCRIPT = textwrap.dedent(
    f"""
    import sys
    sys.path.insert(0, {repr(str(_PROJECT_ROOT))})
    from ui.opportunities_tab import render_opportunities_tab
    render_opportunities_tab()
    """
)


def _collect_opportunities(queue: "Queue") -> None:
    _ensure_real_streamlit_module()
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})
    app = AppTest.from_string(_SCRIPT)
    app.run()
    buttons = app.get("button")
    if not buttons:
        queue.put({"columns": None, "has_df": False})
        return
    buttons[0].click()
    app.run()
    df = opportunities_ctrl.search_opportunities()
    queue.put({
        "columns": list(df.columns),
        "has_df": bool(app.get("arrow_data_frame")),
    })


def _render_app() -> AppTest:
    _ensure_real_streamlit_module()
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})
    app = AppTest.from_string(_SCRIPT)
    app.run()
    return app


def test_header_displays_version() -> None:
    app = _render_app()
    headers = [element.value for element in app.get("header")]
    expected_header = f"🚀 Oportunidades de mercado · versión {__version__}"
    assert expected_header in headers


def test_button_fetches_dataframe_with_spec_columns() -> None:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_collect_opportunities, args=(queue,))
    proc.start()
    proc.join(timeout=15)
    assert proc.exitcode == 0, "Child process running AppTest did not complete successfully"
    result = queue.get()
    columns = result["columns"]
    assert columns is not None
    assert list(columns) == OPPORTUNITIES_SPEC.columns
    assert result["has_df"], "Expected a dataframe element after clicking the button"
