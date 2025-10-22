from __future__ import annotations

import sys
import types

import pytest
import requests

from services import update_checker


class _DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_check_for_update_returns_remote_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_get(url: str, *, timeout: int) -> _DummyResponse:
        calls["url"] = url
        calls["timeout"] = timeout
        return _DummyResponse({"version": "0.5.8"})

    monkeypatch.setattr(update_checker, "_get_local_version", lambda: "0.5.7")
    monkeypatch.setattr(update_checker.requests, "get", fake_get)

    remote = update_checker.check_for_update()

    assert remote == "0.5.8"
    assert calls["url"] == update_checker.REMOTE_VERSION_URL
    assert calls["timeout"] == 3


def test_check_for_update_returns_none_when_versions_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, *, timeout: int) -> _DummyResponse:
        return _DummyResponse({"version": "0.5.7"})

    monkeypatch.setattr(update_checker, "_get_local_version", lambda: "0.5.7")
    monkeypatch.setattr(update_checker.requests, "get", fake_get)

    assert update_checker.check_for_update() is None


def test_check_for_update_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, timeout: int) -> _DummyResponse:
        raise requests.Timeout()

    monkeypatch.setattr(update_checker, "_get_local_version", lambda: "0.5.7")
    monkeypatch.setattr(update_checker.requests, "get", fake_get)

    assert update_checker.check_for_update() is None


def test_run_update_script_executes_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(cmd: list[str], *, check: bool) -> None:
        calls.append((tuple(cmd), {"check": check}))

    fake_streamlit = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        link_button=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(update_checker, "_has_shell_support", lambda: True)
    monkeypatch.setattr(update_checker.subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    result = update_checker._run_update_script("0.5.8")

    assert result is True
    assert [call[0] for call in calls] == [
        ("git", "pull"),
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
            "--upgrade",
        ),
    ]
    assert all(call[1]["check"] for call in calls)
