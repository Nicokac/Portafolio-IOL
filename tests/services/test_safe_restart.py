from __future__ import annotations

import pytest

from services import update_checker


def test_safe_restart_app_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    logs: list[tuple[str, str]] = []
    popen_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    exit_calls: list[int] = []

    def fake_log(event: str, version: str, status: str) -> None:
        logs.append((event, status))

    def fake_popen(cmd: list[str], *, close_fds: bool) -> None:
        popen_calls.append((tuple(cmd), {"close_fds": close_fds}))

    def fake_exit(code: int) -> None:
        exit_calls.append(code)
        raise SystemExit()

    monkeypatch.delenv("DISABLE_AUTO_RESTART", raising=False)
    monkeypatch.setattr(update_checker, "_log_event", fake_log)
    monkeypatch.setattr(update_checker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(update_checker.sys, "exit", fake_exit)
    monkeypatch.setattr(update_checker.sys, "executable", "/usr/bin/python")
    monkeypatch.setattr(update_checker.sys, "argv", ["/app/app.py"])

    with pytest.raises(SystemExit):
        update_checker.safe_restart_app()

    assert popen_calls == [(("/usr/bin/python", "/app/app.py"), {"close_fds": True})]
    assert exit_calls == [0]
    assert ("restart", "initiated") in logs
    assert ("restart", "done") in logs


def test_safe_restart_app_respects_disable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    logs: list[tuple[str, str]] = []

    def fake_log(event: str, version: str, status: str) -> None:
        logs.append((event, status))

    monkeypatch.setenv("DISABLE_AUTO_RESTART", "1")
    monkeypatch.setattr(update_checker, "_log_event", fake_log)

    result = update_checker.safe_restart_app()

    assert result is False
    assert logs == [("restart", "skipped: disabled")]
