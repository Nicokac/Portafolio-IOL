import logging
import os

import requests
import streamlit as st
from requests.exceptions import RequestException

from application.auth_service import get_auth_provider
from application.login_service import clear_password_keys, validate_tokens_key
from services.update_checker import (
    check_for_update,
    format_last_check,
    get_last_check_time,
    _run_update_script,
    get_update_history,
    safe_restart_app,
)
from ui.footer import render_footer
from ui.header import render_header
from ui.helpers.navigation import safe_page_link
from services.auth import generate_token
from ui.security_info import render_security_info
from ui.panels.about import render_about_panel
from ui.panels.diagnostics import render_diagnostics_panel
from shared.config import settings  # Re-exported for backwards compatibility
from shared.errors import AppError, InvalidCredentialsError, NetworkError
from shared.version import __version__


FASTAPI_HEALTH_URL = os.environ.get("FASTAPI_HEALTH_URL", "http://localhost:8000/health")
FASTAPI_ENGINE_INFO_URL = os.environ.get(
    "FASTAPI_ENGINE_INFO_URL", "http://localhost:8000/engine/info"
)


logger = logging.getLogger(__name__)

AUTO_RESTART_KEY = "auto_restart_after_update"
UPDATE_CONFIRM_KEY = "confirm_update_request"
FORCE_CONFIRM_KEY = "confirm_force_update_request"


def _perform_backend_get(
    url: str,
) -> tuple[int | None, dict[str, object] | None]:
    """Execute a GET request against the backend including the auth token if present."""

    try:
        headers = {}
        token = st.session_state.get("auth_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=0.75)
    except RequestException:
        return None, None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return response.status_code, payload


def _is_fastapi_available(url: str = FASTAPI_HEALTH_URL) -> bool:
    """Check if the FastAPI backend is reachable."""

    status_code, payload = _perform_backend_get(url)
    if status_code != 200:
        return False
    if not payload:
        return True
    status = str(payload.get("status", "")).strip().lower()
    return status == "ok" or bool(payload)


def _is_engine_api_active(url: str = FASTAPI_ENGINE_INFO_URL) -> bool:
    """Return ``True`` when the predictive engine endpoint responds successfully."""

    status_code, payload = _perform_backend_get(url)
    if status_code != 200 or not isinstance(payload, dict):
        return False
    status = str(payload.get("status", "")).strip().lower()
    return status == "ok"


def _get_auth_token_ttl() -> int:
    """Return the configured TTL for API tokens, falling back to one hour."""

    raw_value = os.environ.get("FASTAPI_AUTH_TTL", "3600")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 3600
    return max(1, value)


def _ensure_auto_restart_default() -> bool:
    auto_restart_disabled = os.environ.get("DISABLE_AUTO_RESTART") == "1"
    default_choice = not auto_restart_disabled
    if AUTO_RESTART_KEY not in st.session_state:
        st.session_state[AUTO_RESTART_KEY] = default_choice
    if auto_restart_disabled:
        st.session_state[AUTO_RESTART_KEY] = False
    return bool(st.session_state.get(AUTO_RESTART_KEY, default_choice))


def _render_auto_restart_control() -> bool:
    auto_restart_disabled = os.environ.get("DISABLE_AUTO_RESTART") == "1"
    choice = _ensure_auto_restart_default()
    auto_restart = st.checkbox(
        "Reiniciar automáticamente tras la actualización",
        value=choice,
        key=AUTO_RESTART_KEY,
        disabled=auto_restart_disabled,
    )
    if auto_restart_disabled:
        st.info(
            "El reinicio automático está deshabilitado mediante la variable de entorno ``DISABLE_AUTO_RESTART``."
        )
    return auto_restart and not auto_restart_disabled


def _handle_restart(auto_restart_enabled: bool) -> None:
    if auto_restart_enabled:
        st.info("🔁 Reiniciando aplicación...")
        safe_restart_app()
    else:
        st.markdown(
            "<span style='display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 0.8rem;border-radius:999px;background:rgba(30, 136, 229, 0.15);color:#1e88e5;font-weight:600;'>🕓 Reinicio programado</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Podés reiniciar manualmente la aplicación más adelante desde tu entorno de despliegue."
        )


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header()

    if _is_fastapi_available():
        st.markdown(
            """
            <div style="margin:0.5rem 0 1rem;">
                <span style="display:inline-flex;align-items:center;gap:0.45rem;padding:0.45rem 0.95rem;border-radius:999px;background:rgba(46, 125, 50, 0.16);color:#1b5e20;font-weight:600;letter-spacing:0.02em;">
                    🟢 API mode available
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if _is_engine_api_active():
            st.markdown(
                """
                <div style="margin:-0.5rem 0 1.25rem;">
                    <span style="display:inline-flex;align-items:center;gap:0.45rem;padding:0.45rem 1rem;border-radius:999px;background:rgba(123, 31, 162, 0.16);color:#4a148c;font-weight:600;letter-spacing:0.02em;">
                        Engine API active 🔮
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.sidebar:
        safe_page_link(
            "ui.panels.about",
            label="ℹ️ Acerca de",
            render_fallback=render_about_panel,
        )
        safe_page_link(
            "ui.panels.diagnostics",
            label="🩺 Diagnóstico",
            render_fallback=render_diagnostics_panel,
        )

    latest = check_for_update()
    last_check = get_last_check_time()
    last_str = format_last_check(last_check)
    history = get_update_history()

    st.markdown(
        """
        <div style="margin:1rem 0 1.5rem; padding:1rem 1.2rem; border-radius:0.75rem; background:linear-gradient(135deg, rgba(0, 76, 255, 0.08), rgba(0, 199, 190, 0.08));">
            <div style="display:flex; gap:0.8rem; align-items:flex-start;">
                <span style="font-size:1.8rem;">🛰️</span>
                <div>
                    <div style="font-size:1.1rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Observabilidad operativa</div>
                    <p style="margin:0.4rem 0 0; line-height:1.4; color:#333;">
                        Supervisá la salud de tu portafolio con indicadores en tiempo real sin comprometer tus credenciales.
                    </p>
                    <div style="display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.6rem; font-size:0.9rem;">
                        <span style="background:rgba(0,0,0,0.05); padding:0.25rem 0.6rem; border-radius:999px;">⚡ Actualización en vivo</span>
                        <span style="background:rgba(0,0,0,0.05); padding:0.25rem 0.6rem; border-radius:999px;">📊 Consolidación multi-cuenta</span>
                        <span style="background:rgba(0,0,0,0.05); padding:0.25rem 0.6rem; border-radius:999px;">🔍 Trazabilidad completa</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auto_restart_enabled = _render_auto_restart_control()

    if latest:
        st.warning(
            f"💡 Nueva versión disponible: v{latest} (actual: v{__version__})"
        )
        st.link_button(
            "📄 Ver cambios en GitHub",
            "https://github.com/Nicokac/portafolio-iol/blob/main/CHANGELOG.md",
        )
        if st.button("Actualizar ahora"):
            st.session_state[UPDATE_CONFIRM_KEY] = True

        if st.session_state.get(UPDATE_CONFIRM_KEY):
            st.warning("Confirmá la actualización para aplicar los cambios.")
            if st.button("Confirmar actualización"):
                st.status("Actualizando aplicación...", state="running")
                success = _run_update_script(latest)
                st.status("Actualización completada", state="complete")
                if success:
                    st.success("✅ Actualización completada. Reiniciando...")
                    _handle_restart(auto_restart_enabled)
                else:
                    st.error("❌ Ocurrió un error durante la actualización.")
                st.session_state.pop(UPDATE_CONFIRM_KEY, None)
                st.stop()
            if st.button("Cancelar actualización"):
                st.session_state.pop(UPDATE_CONFIRM_KEY, None)

        st.caption(f"Última verificación: {last_str}")
    else:
        try:
            st.status(f"Versión actualizada · v{__version__}", state="complete")
        except Exception:
            st.caption(f"Versión actualizada · v{__version__} ✓")

        st.caption(f"Última verificación: {last_str}")
        st.link_button(
            "📄 Ver cambios en GitHub",
            "https://github.com/Nicokac/portafolio-iol/blob/main/CHANGELOG.md",
        )

        with st.expander("📜 Historial de actualizaciones recientes"):
            if history:
                for entry in reversed(history):
                    st.caption(
                        f"🕒 {entry['timestamp']} — {entry['event']} v{entry['version']} ({entry['status']})"
                    )
            else:
                st.caption("No hay registros previos de actualización.")

    with st.expander("⚙️ Opciones avanzadas"):
        if st.button("Forzar actualización"):
            st.session_state[FORCE_CONFIRM_KEY] = True
        if st.session_state.get(FORCE_CONFIRM_KEY):
            st.warning("Esta acción reinstalará la app desde el repositorio remoto.")
            if st.button("Confirmar actualización"):
                st.status("Actualizando aplicación...", state="running")
                success = _run_update_script(__version__)
                st.status("Actualización completada", state="complete")
                if success:
                    st.success("✅ Actualización completada. Reiniciando...")
                    _handle_restart(auto_restart_enabled)
                else:
                    st.error("❌ Ocurrió un error durante la actualización.")
                st.session_state.pop(FORCE_CONFIRM_KEY, None)
                st.stop()
            if st.button("Cancelar actualización"):
                st.session_state.pop(FORCE_CONFIRM_KEY, None)

    validation = validate_tokens_key()
    if validation.message:
        if validation.level == "warning":
            st.warning(validation.message)
        elif validation.level == "error":
            st.error(validation.message)
    if not validation.can_proceed:
        return

    err = st.session_state.pop("login_error", "")
    if err:
        st.error(err)

    form_col, info_col = st.columns([2.5, 1.5])

    with form_col:
        with st.form("login_form"):
            user = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Iniciar sesión")

    with info_col:
        render_security_info()

    render_footer()

    if submitted:
        provider = get_auth_provider()

        try:
            provider.login(user, password)
            st.session_state["auth_token"] = generate_token(
                user,
                _get_auth_token_ttl(),
            )
            st.session_state["authenticated"] = True
            st.session_state.pop("force_login", None)
            st.session_state.pop("login_error", None)
            clear_password_keys(st.session_state)
            st.rerun()
        except InvalidCredentialsError:
            logger.warning("Fallo de autenticación")
            st.session_state["login_error"] = "Usuario o contraseña inválidos"
            clear_password_keys(st.session_state)
            st.rerun()
        except NetworkError:
            logger.warning("Error de conexión durante el login")
            st.session_state["login_error"] = "Error de conexión"
            clear_password_keys(st.session_state)
            st.rerun()
        except AppError as err:
            logger.warning("Error controlado durante el login: %s", err)
            msg = str(err)
            st.error(msg)
            st.session_state["login_error"] = msg
            clear_password_keys(st.session_state)
            st.rerun()
        except RuntimeError as e:
            logger.exception("Error durante el login: %s", e)
            st.session_state["login_error"] = str(e)
            clear_password_keys(st.session_state)
            st.rerun()
        except Exception:
            logger.exception("Error inesperado durante el login")
            st.session_state["login_error"] = "Error inesperado, contacte soporte"
            clear_password_keys(st.session_state)
            st.rerun()


__all__ = ["render_login_page", "settings"]
