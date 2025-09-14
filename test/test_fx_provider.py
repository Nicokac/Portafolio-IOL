import json
import time
from unittest.mock import MagicMock

import requests

from infrastructure.fx import provider as fx_provider


# _load_cache tests

def test_load_cache_expired_file(monkeypatch, tmp_path):
    cache_file = tmp_path / "fx_cache.json"
    data = {"oficial": 1, "_ts": time.time() - fx_provider.CACHE_TTL - 10}
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(fx_provider, "CACHE_FILE", cache_file)
    provider = fx_provider.FXProviderAdapter()
    assert provider._load_cache() is None


def test_load_cache_invalid_json(monkeypatch, tmp_path):
    cache_file = tmp_path / "fx_cache.json"
    cache_file.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(fx_provider, "CACHE_FILE", cache_file)
    provider = fx_provider.FXProviderAdapter()
    assert provider._load_cache() is None


# _save_cache tests

def test_save_cache_handles_exception(monkeypatch, tmp_path):
    cache_file = tmp_path / "fx_cache.json"
    monkeypatch.setattr(fx_provider, "CACHE_FILE", cache_file)
    provider = fx_provider.FXProviderAdapter()

    from pathlib import Path

    def bad_write(self, *args, **kwargs):
        raise PermissionError("no write")

    monkeypatch.setattr(Path, "write_text", bad_write)

    provider._save_cache({"oficial": 1, "_ts": 1})
    assert not cache_file.exists()


# _load_fallback tests

def test_load_fallback_returns_data(monkeypatch, tmp_path):
    fallback_file = tmp_path / "fallback.json"
    fallback_file.write_text(json.dumps({"oficial": "2"}), encoding="utf-8")
    monkeypatch.setattr(fx_provider, "FALLBACK_FILE", fallback_file)
    provider = fx_provider.FXProviderAdapter()
    data = provider._load_fallback()
    assert data and data["oficial"] == 2.0


# get_rates tests

def test_get_rates_request_exception_uses_fallback(monkeypatch):
    provider = fx_provider.FXProviderAdapter()

    def raise_exc(url):
        raise requests.RequestException("boom")

    provider.session.get = MagicMock(side_effect=raise_exc)
    fallback = {"oficial": 1.0, "_ts": int(time.time())}
    monkeypatch.setattr(provider, "_load_cache", lambda: None)
    monkeypatch.setattr(provider, "_load_fallback", lambda: fallback)
    monkeypatch.setattr(provider, "_save_cache", lambda data: None)

    rates, err = provider.get_rates()
    assert rates == fallback
    assert err and "Usando datos locales de FX" in err


def test_get_rates_general_exception_returns_cache(monkeypatch):
    provider = fx_provider.FXProviderAdapter()
    provider.session.get = MagicMock(side_effect=ValueError("boom"))

    cache = {"oficial": 1.0, "_ts": int(time.time())}
    calls = {"n": 0}

    def fake_load_cache():
        if calls["n"] == 0:
            calls["n"] += 1
            return None
        return cache

    monkeypatch.setattr(provider, "_load_cache", fake_load_cache)
    monkeypatch.setattr(provider, "_load_fallback", lambda: None)
    monkeypatch.setattr(provider, "_save_cache", lambda data: None)

    rates, err = provider.get_rates()
    assert rates == cache
    assert err and "FXProviderAdapter failed" in err


def test_get_rates_general_exception_returns_fallback(monkeypatch):
    provider = fx_provider.FXProviderAdapter()
    provider.session.get = MagicMock(side_effect=ValueError("boom"))

    fallback = {"blue": 2.0, "_ts": int(time.time())}
    monkeypatch.setattr(provider, "_load_cache", lambda: None)
    monkeypatch.setattr(provider, "_load_fallback", lambda: fallback)
    monkeypatch.setattr(provider, "_save_cache", lambda data: None)

    rates, err = provider.get_rates()
    assert rates == fallback
    assert err and "FXProviderAdapter failed" in err