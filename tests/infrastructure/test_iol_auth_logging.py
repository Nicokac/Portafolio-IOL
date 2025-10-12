"""Ensure sensitive information is not leaked in IOL auth logs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

from infrastructure.iol import auth as iol_auth


@pytest.fixture(autouse=True)
def _prepare_fernet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iol_auth, "FERNET", Fernet(Fernet.generate_key()))


def _fake_response(status_code: int) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status_code,
        text="sensitive-body",
        json=lambda: {},
        raise_for_status=lambda: None,
    )


def test_login_failure_logs_without_payload(monkeypatch: pytest.MonkeyPatch, caplog, tmp_path):
    auth = iol_auth.IOLAuth("user", "pass", tokens_file=tmp_path / "tokens.json")
    monkeypatch.setattr(auth.session, "post", lambda *_, **__: _fake_response(401))

    caplog.set_level("WARNING", logger=iol_auth.__name__)
    with pytest.raises(iol_auth.InvalidCredentialsError):
        auth.login()

    assert "Auth failed (code=401)" in caplog.text
    assert "sensitive-body" not in caplog.text


def test_refresh_failure_logs_without_payload(monkeypatch: pytest.MonkeyPatch, caplog, tmp_path):
    auth = iol_auth.IOLAuth("user", "pass", tokens_file=tmp_path / "tokens.json")
    auth.tokens = {"refresh_token": "token"}
    monkeypatch.setattr(auth.session, "post", lambda *_, **__: _fake_response(400))

    caplog.set_level("WARNING", logger=iol_auth.__name__)
    with pytest.raises(iol_auth.InvalidCredentialsError):
        auth.refresh()

    assert "Auth failed (code=400)" in caplog.text
    assert "sensitive-body" not in caplog.text
