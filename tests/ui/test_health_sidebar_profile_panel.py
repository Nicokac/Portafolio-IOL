"""Tests for the investor profile panel rendered in the health sidebar."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import ui.health_sidebar as health_sidebar_module


@pytest.fixture
def profile_sidebar(monkeypatch: pytest.MonkeyPatch, streamlit_stub) -> SimpleNamespace:
    streamlit_stub.reset()
    module = importlib.reload(health_sidebar_module)
    monkeypatch.setattr(module, "st", streamlit_stub)
    return SimpleNamespace(module=module, streamlit=streamlit_stub)


def test_profile_section_renders_details(profile_sidebar: SimpleNamespace) -> None:
    module = profile_sidebar.module
    st = profile_sidebar.streamlit
    st.session_state["iol_user_profile"] = {
        "nombre": "Juan Pérez",
        "perfil_inversor": "Moderado",
        "preferencias": ["CEDEARs Diversificados", "FCI Balanceados"],
    }

    module._render_investor_profile_section(st.sidebar)

    markdowns = st.sidebar.markdowns
    assert "#### 👤 Perfil del inversor" in markdowns
    assert "👤 **Nombre:** Juan Pérez" in markdowns
    assert "📊 **Perfil inversor:** Moderado" in markdowns
    assert "💡 **Preferencias:** CEDEARs Diversificados, FCI Balanceados" in markdowns


def test_profile_section_handles_missing_profile(profile_sidebar: SimpleNamespace) -> None:
    module = profile_sidebar.module
    st = profile_sidebar.streamlit

    module._render_investor_profile_section(st.sidebar)

    assert "_Perfil del inversor no disponible._" in st.sidebar.markdowns


def test_profile_section_refresh_button(profile_sidebar: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    module = profile_sidebar.module
    st = profile_sidebar.streamlit
    st.session_state["iol_user_profile"] = {"nombre": "Ana"}
    st.set_button_result("🔄 Actualizar perfil", True)

    cleared: list[bool] = []
    monkeypatch.setattr(module.st, "cache_data", SimpleNamespace(clear=lambda: cleared.append(True)), raising=False)

    module._render_investor_profile_section(st.sidebar)

    assert cleared == [True]
    assert st.session_state.get("_profile_refresh_pending") is True
    assert "iol_user_profile" not in st.session_state
