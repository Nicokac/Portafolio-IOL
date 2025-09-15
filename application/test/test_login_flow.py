import streamlit as st
from ui.actions import render_action_menu
from unittest.mock import patch


class DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def button(self, *args, **kwargs):
        return False


def test_logout_forces_login_page(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    st.session_state["IOL_USERNAME"] = "user"
    st.session_state["authenticated"] = True
    st.session_state["logout_pending"] = True

    monkeypatch.setattr(st, "popover", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "columns", lambda n: [DummyCtx(), DummyCtx()])
    monkeypatch.setattr(st, "spinner", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "toast", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(st, "stop", lambda *a, **k: None)
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)

    with patch("ui.actions.auth_service.logout") as mock_logout:
        def fake_logout(*a, **k):
            st.session_state.clear()
            st.session_state["force_login"] = True

        mock_logout.side_effect = fake_logout
        render_action_menu()
        mock_logout.assert_called_once_with("user")

    assert st.session_state.get("force_login") is True

    with patch("ui.login.render_login_page") as mock_login:
        from ui.login import render_login_page
        if st.session_state.get("force_login"):
            render_login_page()
        mock_login.assert_called_once()

    st.session_state.clear()

def test_service_build_iol_client_returns_error(monkeypatch):
    from services import cache as svc_cache

    monkeypatch.setattr(st, "session_state", {})
    st.session_state["IOL_USERNAME"] = "user"

    def boom(cache_key, user, tokens_file):
        raise RuntimeError("bad creds")

    monkeypatch.setattr(svc_cache, "get_client_cached", boom)

    cli, error = svc_cache.build_iol_client()

    assert cli is None
    assert str(error) == "bad creds"


def test_controller_build_iol_client_handles_error(monkeypatch):
    from controllers import auth

    monkeypatch.setattr(st, "session_state", {})

    class DummyProvider:
        def build_client(self):
            return None, RuntimeError("bad creds")

    monkeypatch.setattr(auth, "get_auth_provider", lambda: DummyProvider())
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)

    auth.build_iol_client()

    assert st.session_state.get("force_login") is True
    assert st.session_state.get("login_error") == "Error de conexión"
    assert "IOL_PASSWORD" not in st.session_state


def test_controller_build_iol_client_success_clears_password(monkeypatch):
    from controllers import auth

    monkeypatch.setattr(st, "session_state", {})

    dummy_cli = object()

    class DummyProvider:
        def build_client(self):
            return dummy_cli, None

    monkeypatch.setattr(auth, "get_auth_provider", lambda: DummyProvider())

    cli = auth.build_iol_client()

    assert cli is dummy_cli
    assert st.session_state.get("authenticated") is True
    assert "IOL_PASSWORD" not in st.session_state


def test_render_login_page_shows_error(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    st.session_state["login_error"] = "fail"

    monkeypatch.setattr("ui.login.render_header", lambda: None)
    monkeypatch.setattr(st, "form", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "text_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **k: False)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)

    from shared import config
    config.settings.tokens_key = "k"

    captured = {}
    monkeypatch.setattr(st, "error", lambda msg: captured.setdefault("msg", msg))

    from ui.login import render_login_page

    render_login_page()

    assert captured.get("msg") == "fail"


def _prepare_login_form(monkeypatch, submitted=True):
    monkeypatch.setattr("ui.login.render_header", lambda: None)
    monkeypatch.setattr(st, "form", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "text_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **k: submitted)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)


def test_render_login_page_handles_invalid_credentials(monkeypatch):
    from ui import login

    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)

    class DummyProvider:
        def login(self, u, p):
            raise login.InvalidCredentialsError()

    monkeypatch.setattr(login, "get_auth_provider", lambda: DummyProvider())

    login.render_login_page()

    assert st.session_state.get("login_error") == "Usuario o contraseña inválidos"
    assert not any("password" in k.lower() for k in st.session_state)


def test_render_login_page_handles_network_error(monkeypatch):
    from ui import login

    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)

    class DummyProvider:
        def login(self, u, p):
            raise login.NetworkError()

    monkeypatch.setattr(login, "get_auth_provider", lambda: DummyProvider())

    login.render_login_page()

    assert st.session_state.get("login_error") == "Error de conexión"
    assert not any("password" in k.lower() for k in st.session_state)


def test_render_login_page_handles_timeout(monkeypatch):
    from ui import login

    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)

    class DummyProvider:
        def login(self, u, p):
            raise login.TimeoutError()

    monkeypatch.setattr(login, "get_auth_provider", lambda: DummyProvider())

    login.render_login_page()

    assert st.session_state.get("login_error") == "Tiempo de espera agotado"
    assert not any("password" in k.lower() for k in st.session_state)


def test_render_login_page_handles_tokens_key_missing(monkeypatch):
    from ui import login

    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)

    class DummyProvider:
        def login(self, u, p):
            raise RuntimeError("no key")

    monkeypatch.setattr(login, "get_auth_provider", lambda: DummyProvider())

    login.render_login_page()

    assert st.session_state.get("login_error") == "no key"
    assert not any("password" in k.lower() for k in st.session_state)

def test_login_success_creates_tokens_file_and_clears_password(monkeypatch, tmp_path):
    from ui import login
    from controllers import auth as ctrl_auth
    from application import auth_service
    from shared.cache import cache as app_cache

    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)
    monkeypatch.setattr(login.settings, "tokens_key", "k")

    tokens_path = tmp_path / "tok.json"
    dummy_cli = object()

    class DummyAuth:
        def login(self, u, p):
            tokens_path.write_text("{}")
            app_cache.set("tokens_file", str(tokens_path))
            return {"access_token": "tok"}

        def logout(self, u, p=""):
            pass

        def build_client(self):
            return dummy_cli, None

    provider = DummyAuth()
    monkeypatch.setattr(auth_service, "_provider", provider)

    login.render_login_page()

    assert tokens_path.exists()

    cli = ctrl_auth.build_iol_client()

    assert cli is dummy_cli
    assert not any("password" in k.lower() for k in st.session_state)


def test_login_invalid_credentials_no_tokens_file(monkeypatch, tmp_path):
    from ui import login
    from application import auth_service

    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)
    monkeypatch.setattr(login.settings, "tokens_key", "k")

    tokens_path = tmp_path / "tok.json"

    class DummyAuth:
        def login(self, u, p):
            raise login.InvalidCredentialsError()

        def logout(self, u, p=""):
            pass

        def build_client(self):
            return object(), None

    provider = DummyAuth()
    monkeypatch.setattr(auth_service, "_provider", provider)

    login.render_login_page()

    assert not tokens_path.exists()
    assert st.session_state.get("login_error") == "Usuario o contraseña inválidos"
    assert not any("password" in k.lower() for k in st.session_state)


def test_rerun_reuses_token_without_password(monkeypatch, tmp_path):
    from ui import login
    from controllers import auth as ctrl_auth
    from application import auth_service
    from services import cache as svc_cache
    from shared.cache import cache as app_cache

    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)
    monkeypatch.setattr(login.settings, "tokens_key", "k")

    tokens_path = tmp_path / "tok.json"
    dummy_cli = object()
    calls: list[str] = []

    def fake_get_client_cached(cache_key, user, tokens_file):
        calls.append(tokens_file)
        return dummy_cli

    monkeypatch.setattr(svc_cache, "get_client_cached", fake_get_client_cached)

    login_calls = []

    class DummyAuth:
        def login(self, u, p):
            login_calls.append((u, p))
            tokens_path.write_text("{}")
            app_cache.set("tokens_file", str(tokens_path))
            return {"access_token": "tok"}

        def logout(self, u, p=""):
            pass

        def build_client(self):
            return svc_cache.build_iol_client()

    provider = DummyAuth()
    monkeypatch.setattr(auth_service, "_provider", provider)

    login.render_login_page()
    cli1 = ctrl_auth.build_iol_client()
    assert cli1 is dummy_cli
    assert not any("password" in k.lower() for k in st.session_state)

    cli2 = ctrl_auth.build_iol_client()
    assert cli2 is dummy_cli
    assert not any("password" in k.lower() for k in st.session_state)

    assert calls == [str(tokens_path), str(tokens_path)]
    assert len(login_calls) == 1
    assert tokens_path.exists()


def test_full_login_logout_relogin_clears_password(monkeypatch, tmp_path):
    from ui import login
    from controllers import auth as ctrl_auth
    from application import auth_service
    from services import cache as svc_cache
    from shared.cache import cache as app_cache

    # simulate initial login
    # simulate initial login
    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "u", "some_password": "p"})
    _prepare_login_form(monkeypatch)
    monkeypatch.setattr(login.settings, "tokens_key", "k")

    tokens_path = tmp_path / "tok.json"
    dummy_cli = object()
    calls: list[str] = []

    def fake_get_client_cached(cache_key, user, tokens_file):
        calls.append(tokens_file)
        return dummy_cli

    monkeypatch.setattr(svc_cache, "get_client_cached", fake_get_client_cached)

    class DummyAuth:
        def login(self, u, p):
            tokens_path.write_text("{}")
            app_cache.set("tokens_file", str(tokens_path))
            return {"access_token": "tok"}

        def logout(self, u, p=""):
            for key in ("IOL_USERNAME", "IOL_PASSWORD", "authenticated", "client_salt"):
                st.session_state.pop(key, None)
            app_cache.pop("tokens_file", None)

        def build_client(self):
            return svc_cache.build_iol_client()

    monkeypatch.setattr(auth_service, "_provider", DummyAuth())

    # login removes password
    login.render_login_page()
    assert not any("password" in k.lower() for k in st.session_state)

    # build client uses token without password
    cli1 = ctrl_auth.build_iol_client()
    assert cli1 is dummy_cli
    assert calls == [str(tokens_path)]
    assert not any("password" in k.lower() for k in st.session_state)

    # logout clears all credentials
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    auth_service.logout("u")
    assert "IOL_USERNAME" not in st.session_state
    assert not any("password" in k.lower() for k in st.session_state)
    assert "authenticated" not in st.session_state
    assert app_cache.get("tokens_file") is None

    # relogin should not leave password in memory
    st.session_state["IOL_USERNAME"] = "u"
    st.session_state["some_password"] = "p2"
    _prepare_login_form(monkeypatch)
    login.render_login_page()
    assert not any("password" in k.lower() for k in st.session_state)

    cli2 = ctrl_auth.build_iol_client()
    assert cli2 is dummy_cli
    assert calls == [str(tokens_path), str(tokens_path)]
    assert not any("password" in k.lower() for k in st.session_state)
