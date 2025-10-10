import logging
import streamlit as st
from application.auth_service import get_auth_provider
from application.login_service import clear_password_keys, validate_tokens_key
from services.update_checker import check_for_update, _run_update_script
from ui.footer import render_footer
from ui.header import render_header
from ui.security_info import render_security_info
from shared.config import settings  # Re-exported for backwards compatibility
from shared.errors import AppError, InvalidCredentialsError, NetworkError
from shared.version import __version__


logger = logging.getLogger(__name__)


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header()

    latest = check_for_update()

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

    if latest:
        st.warning(f"Nueva versión disponible: v{latest} (actual: v{__version__})")
        if st.button("Actualizar ahora"):
            st.info("Iniciando actualización...")
            if _run_update_script(latest):
                st.success("Actualización completada. Reinicie la aplicación.")
                st.stop()
    else:
        st.caption(f"Versión actual: v{__version__}")

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
