import types
from unittest.mock import Mock
import pytest

from ui import login


class DummyForm:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def make_st(submitted: bool = False):
    st_mock = types.SimpleNamespace()
    st_mock.session_state = {}
    st_mock.warning = Mock()
    st_mock.error = Mock()
    st_mock.caption = lambda *a, **k: None
    st_mock.text_input = lambda *a, **k: ""
    st_mock.form_submit_button = Mock(return_value=submitted)
    st_mock.form = lambda *a, **k: DummyForm()
    st_mock.rerun = Mock()
    return st_mock


def test_tokens_key_missing_warns_when_plain_allowed(monkeypatch):
    st_mock = make_st(submitted=False)
    monkeypatch.setattr(login, "st", st_mock)
    monkeypatch.setattr(login, "render_header", lambda: None)
    monkeypatch.setattr(login.settings, "tokens_key", None)
    monkeypatch.setattr(login.settings, "allow_plain_tokens", True)
    monkeypatch.setattr(login, "get_auth_provider", lambda: None)

    login.render_login_page()

    st_mock.warning.assert_called_once()
    st_mock.error.assert_not_called()
    st_mock.form_submit_button.assert_called_once()


def test_tokens_key_missing_errors_when_plain_disallowed(monkeypatch):
    st_mock = make_st(submitted=False)
    st_mock.form = Mock(return_value=DummyForm())
    monkeypatch.setattr(login, "st", st_mock)
    monkeypatch.setattr(login, "render_header", lambda: None)
    monkeypatch.setattr(login.settings, "tokens_key", None)
    monkeypatch.setattr(login.settings, "allow_plain_tokens", False)
    provider = Mock()
    monkeypatch.setattr(login, "get_auth_provider", provider)

    login.render_login_page()

    st_mock.error.assert_called_once()
    st_mock.warning.assert_not_called()
    st_mock.form.assert_not_called()
    provider.assert_not_called()


def _run_login_with_exception(monkeypatch, exc, expected_msg):
    st_mock = make_st(submitted=True)
    st_mock.session_state["some_password"] = "secret"
    monkeypatch.setattr(login, "st", st_mock)
    monkeypatch.setattr(login, "render_header", lambda: None)
    monkeypatch.setattr(login.settings, "tokens_key", "k")

    provider = types.SimpleNamespace()
    def do_login(u, p):
        raise exc
    provider.login = do_login
    monkeypatch.setattr(login, "get_auth_provider", lambda: provider)

    login.render_login_page()

    assert st_mock.session_state.get("login_error") == expected_msg
    st_mock.rerun.assert_called_once()
    assert not any("password" in k.lower() for k in st_mock.session_state)


def test_login_invalid_credentials_sets_error_and_reruns(monkeypatch):
    class DummyInvalid(Exception):
        pass

    monkeypatch.setattr(login, "InvalidCredentialsError", DummyInvalid)
    _run_login_with_exception(monkeypatch, DummyInvalid(), "Usuario o contraseña inválidos")


def test_login_network_error_sets_error_and_reruns(monkeypatch):
    class DummyNet(Exception):
        pass

    monkeypatch.setattr(login, "NetworkError", DummyNet)
    _run_login_with_exception(monkeypatch, DummyNet(), "Error de conexión")


def test_login_runtime_error_sets_error_and_reruns(monkeypatch):
    class DummyRun(Exception):
        pass

    monkeypatch.setattr(login, "RuntimeError", DummyRun, raising=False)
    _run_login_with_exception(monkeypatch, DummyRun("boom"), "boom")


def test_login_unexpected_exception_sets_generic_error_and_reruns(monkeypatch):
    class DummyEx(Exception):
        pass

    monkeypatch.setattr(login, "Exception", DummyEx, raising=False)
    _run_login_with_exception(monkeypatch, DummyEx("oops"), "Error inesperado, contacte soporte")
