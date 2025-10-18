from contextlib import nullcontext
import importlib
import logging
import sys
import types

import pytest


class _StubStreamlit(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state: dict[str, object] = {}

    def rerun(self) -> None:  # pragma: no cover - defensive stub
        raise RuntimeError("rerun not supported in tests")


class DummyProvider:
    def build_client(self):
        return object(), None


def test_build_iol_client_triggers_visual_cache_prewarm(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "controllers.auth",
        "application",
        "application.auth_service",
    ):
        sys.modules.pop(name, None)

    stub_st = _StubStreamlit()
    sys.modules["streamlit"] = stub_st

    stub_services_auth = types.ModuleType("services.auth")
    stub_services_auth.revoke_token = lambda *_args, **_kwargs: None
    sys.modules["services.auth"] = stub_services_auth

    stub_timer = types.ModuleType("services.performance_timer")
    stub_timer.performance_timer = lambda *_, **__: nullcontext()
    sys.modules["services.performance_timer"] = stub_timer

    stub_client = types.ModuleType("infrastructure.iol.client")
    class _DummyIIOLProvider:  # pragma: no cover - simple placeholder
        ...
    stub_client.IIOLProvider = _DummyIIOLProvider
    stub_client.IOLClient = object
    sys.modules["infrastructure.iol.client"] = stub_client

    stub_auth = types.ModuleType("infrastructure.iol.auth")
    class _InvalidCredentialsError(Exception):
        ...
    stub_auth.InvalidCredentialsError = _InvalidCredentialsError
    sys.modules["infrastructure.iol.auth"] = stub_auth

    app_pkg = types.ModuleType("application")
    app_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["application"] = app_pkg
    stub_app_auth = types.ModuleType("application.auth_service")
    stub_app_auth.get_auth_provider = lambda: DummyProvider()
    sys.modules["application.auth_service"] = stub_app_auth

    auth_mod = importlib.import_module("controllers.auth")
    fake_st = stub_st

    try:
        monkeypatch.setattr(auth_mod, "logger", logging.getLogger("controllers.auth.test"))
        monkeypatch.setattr(auth_mod, "_get_performance_timer", lambda: (lambda *_, **__: nullcontext()))
        monkeypatch.setattr(auth_mod, "get_auth_provider", lambda: DummyProvider())

        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            auth_mod,
            "prewarm_visual_cache",
            lambda **kwargs: calls.append(kwargs) or [],
        )

        cli = auth_mod.build_iol_client()

        assert cli is not None
        assert fake_st.session_state.get("authenticated") is True
        assert auth_mod.LOGIN_AUTH_TIMESTAMP_KEY in fake_st.session_state
        assert calls == [{}]
    finally:
        for name in (
            "streamlit",
            "services.auth",
            "services.performance_timer",
            "infrastructure.iol.client",
            "infrastructure.iol.auth",
            "application.auth_service",
            "application",
            "controllers.auth",
        ):
            sys.modules.pop(name, None)
