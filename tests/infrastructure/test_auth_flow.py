"""Authentication bootstrap flow regression tests."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from infrastructure.iol import auth as iol_auth


def _write_encrypted_tokens(path: Path, payload: dict[str, object]) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    content = iol_auth.FERNET.encrypt(raw) if iol_auth.FERNET else raw
    path.write_bytes(content)


@pytest.fixture(autouse=True)
def _prepare_fernet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iol_auth, "FERNET", Fernet(Fernet.generate_key()))


def test_bootstrap_refresh_is_silent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> None:
    tokens_path = tmp_path / "tokens.json"
    stale_tokens = {
        "access_token": "old",
        "refresh_token": "refresh",
        "expires_in": 3600,
        "timestamp": int(time.time() - 40 * 86400),
    }
    _write_encrypted_tokens(tokens_path, stale_tokens)

    called: dict[str, object] = {}

    def _fake_refresh(self: iol_auth.IOLAuth, *, silent: bool = False):
        called["silent"] = silent
        self.tokens = {"access_token": "new", "refresh_token": "refresh"}
        return self.tokens

    monkeypatch.setattr(iol_auth.IOLAuth, "refresh", _fake_refresh, raising=False)

    caplog.set_level(logging.INFO, logger=iol_auth.__name__)
    auth = iol_auth.IOLAuth("user", "pass", tokens_file=tokens_path)

    assert called.get("silent") is True
    assert auth.tokens.get("access_token") == "new"
    assert "Auth failed" not in caplog.text


def test_bootstrap_refresh_clears_invalid_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> None:
    tokens_path = tmp_path / "tokens.json"
    stale_tokens = {
        "access_token": "old",
        "refresh_token": "refresh",
        "timestamp": int(time.time() - 45 * 86400),
    }
    _write_encrypted_tokens(tokens_path, stale_tokens)

    def _failing_refresh(self: iol_auth.IOLAuth, *, silent: bool = False):
        assert silent is True
        raise iol_auth.InvalidCredentialsError("invalid")

    monkeypatch.setattr(iol_auth.IOLAuth, "refresh", _failing_refresh, raising=False)

    caplog.set_level(logging.INFO, logger=iol_auth.__name__)
    auth = iol_auth.IOLAuth("user", "pass", tokens_file=tokens_path)

    assert auth.tokens == {}
    assert not tokens_path.exists()
    assert "Auth failed" not in caplog.text
