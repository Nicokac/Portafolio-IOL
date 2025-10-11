from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import update_checker


class _DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_check_for_update_logs_event(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str, str]] = []

    def fake_log(event: str, version: str, status: str) -> None:
        events.append((event, version, status))

    monkeypatch.setattr(update_checker, "_get_local_version", lambda: "0.5.9")
    monkeypatch.setattr(
        update_checker.requests,
        "get",
        lambda _url, *, timeout: _DummyResponse({"version": "0.5.9"}),
    )
    monkeypatch.setattr(update_checker, "_log_event", fake_log)

    update_checker.check_for_update()

    assert ("check", "0.5.9", "ok") in events


def test_run_update_script_logs_started_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str, str]] = []

    def fake_log(event: str, version: str, status: str) -> None:
        events.append((event, version, status))

    monkeypatch.setattr(update_checker, "_has_shell_support", lambda: True)
    monkeypatch.setattr(update_checker.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(update_checker, "_log_event", fake_log)

    fake_st = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        link_button=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    try:
        result = update_checker._run_update_script("0.5.9")
    finally:
        sys.modules.pop("streamlit", None)

    assert result is True
    assert events == [
        ("update", "0.5.9", "started"),
        ("update", "0.5.9", "done"),
    ]


def test_log_event_persists_last_twenty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "portafolio_iol_update_log.json"
    monkeypatch.setattr(update_checker, "LOG_FILE", str(log_path), raising=False)

    for idx in range(25):
        update_checker._log_event("check", f"{idx}", "ok")

    with log_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert len(payload) == 20
    assert [entry["version"] for entry in payload] == [str(i) for i in range(5, 25)]
