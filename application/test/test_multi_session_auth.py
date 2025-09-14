import re
import hashlib
from pathlib import Path

import streamlit as st

from application import auth_service



def _token_file(user: str) -> Path:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
    user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
    return Path("tokens") / f"{sanitized}-{user_hash}.json"


def test_multi_session_tokens_and_logout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)

    from services import cache as svc_cache

    class DummyCacheResource:
        def clear(self, key=None):
            pass

    dummy = DummyCacheResource()
    monkeypatch.setattr(svc_cache, "get_client_cached", dummy)
    monkeypatch.setattr(svc_cache, "fetch_portfolio", dummy)
    monkeypatch.setattr(svc_cache, "fetch_quotes_bulk", dummy)
    monkeypatch.setattr(svc_cache, "fetch_fx_rates", dummy)

    class DummyAuth:
        def __init__(self, user, password, tokens_file, allow_plain_tokens):
            self.tokens_file = tokens_file

        def login(self):
            p = Path(self.tokens_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{\"access_token\": \"x\"}")
            return {"access_token": "x"}

        def clear_tokens(self):
            Path(self.tokens_file).unlink(missing_ok=True)

    monkeypatch.setattr(auth_service, "IOLAuth", DummyAuth)

    session_a: dict = {}
    session_b: dict = {}

    # User A login
    monkeypatch.setattr(st, "session_state", session_a)
    auth_service.login("userA", "pw")
    session_a["authenticated"] = True
    file_a = _token_file("userA")
    content_a = file_a.read_text()
    assert file_a.exists()

    # User B login
    monkeypatch.setattr(st, "session_state", session_b)
    auth_service.login("userB", "pw")
    session_b["authenticated"] = True
    file_b = _token_file("userB")
    assert file_b.exists()

    # Both tokens exist and A's token is unchanged
    assert file_a.exists()
    assert file_a.read_text() == content_a
    assert file_a != file_b

    # Logout user A
    monkeypatch.setattr(st, "session_state", session_a)
    auth_service.logout("userA")
    assert not file_a.exists()
    assert session_a.get("authenticated") is None

    # User B remains authenticated and token persists
    monkeypatch.setattr(st, "session_state", session_b)
    assert file_b.exists()
    assert session_b.get("authenticated") is True
