import re
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_login_and_logout_sanitize_token_path(monkeypatch):
    from application import auth_service as auth

    captured = {}

    class DummyAuth:
        def __init__(self, user, password, tokens_file):
            captured.setdefault('paths', []).append(tokens_file)

        def login(self):
            return {'access_token': 'x'}

        def clear_tokens(self):
            captured['cleared'] = True

    monkeypatch.setattr(auth, 'IOLAuth', DummyAuth)

    class DummyCache:
        def set(self, key, value):
            captured[key] = value

    monkeypatch.setattr(auth, 'cache', DummyCache())

    user = "us$er*#name"
    provider = auth.IOLAuthenticationProvider()
    provider.login(user, 'p')
    provider.logout(user, 'p')

    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
    expected = Path('tokens') / f"{sanitized}.json"
    assert captured['paths'][0] == expected
    assert captured['paths'][1] == expected


def test_build_iol_client_sanitizes_token_path(monkeypatch):
    from services import cache as cache_module
    from shared import cache as shared_cache

    st = SimpleNamespace(session_state={})
    st.session_state.update({'IOL_USERNAME': 'ab?c', 'IOL_PASSWORD': 'p'})
    monkeypatch.setattr(cache_module, 'st', st)
    monkeypatch.setattr(shared_cache, 'st', st)

    captured = {}

    def dummy_get_client_cached(cache_key, user, password, tokens_file):
        captured['tokens_file'] = tokens_file
        class DummyClient:
            pass
        return DummyClient()

    monkeypatch.setattr(cache_module, 'get_client_cached', dummy_get_client_cached)

    cache_module.cache.pop('tokens_file', None)

    cli, err = cache_module.build_iol_client()
    assert err is None
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", 'ab?c')
    expected = Path('tokens') / f"{sanitized}.json"
    assert captured['tokens_file'] == expected
