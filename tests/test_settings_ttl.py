"""Tests for cache TTL exports sourced from shared.settings."""
from __future__ import annotations

import importlib
from typing import Iterable

import pytest

TTL_EXPORTS: tuple[str, ...] = (
    "cache_ttl_portfolio",
    "cache_ttl_last_price",
    "cache_ttl_fx",
    "cache_ttl_quotes",
    "quotes_hist_maxlen",
    "max_quote_workers",
)

ENV_KEYS: tuple[str, ...] = (
    "CACHE_TTL_PORTFOLIO",
    "CACHE_TTL_LAST_PRICE",
    "CACHE_TTL_FX",
    "CACHE_TTL_QUOTES",
    "QUOTES_HIST_MAXLEN",
    "MAX_QUOTE_WORKERS",
)


@pytest.fixture(autouse=True)
def reload_settings_after_test():
    """Ensure shared configuration modules are reset after each test."""
    yield
    _reload_settings_module()


def _reload_settings_module():
    """Reload configuration modules and return the shared.settings module."""
    shared_config = importlib.import_module("shared.config")
    shared_settings = importlib.import_module("shared.settings")
    importlib.reload(shared_config)
    importlib.reload(shared_settings)
    return shared_settings


def _clear_env(monkeypatch: pytest.MonkeyPatch, keys: Iterable[str]) -> None:
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_ttl_exports_match_underlying_settings(monkeypatch: pytest.MonkeyPatch):
    """Exported TTL constants should mirror the values in the Settings object."""
    _clear_env(monkeypatch, ENV_KEYS)
    settings_module = _reload_settings_module()
    for attr in TTL_EXPORTS:
        exported = getattr(settings_module, attr)
        underlying = getattr(settings_module.settings, attr)
        assert exported == underlying, attr


def test_ttl_values_respect_environment_overrides(monkeypatch: pytest.MonkeyPatch):
    """Environment variables must override the default TTL configuration."""
    overrides = {
        "CACHE_TTL_PORTFOLIO": "123",
        "CACHE_TTL_LAST_PRICE": "45",
        "CACHE_TTL_FX": "67",
        "CACHE_TTL_QUOTES": "89",
        "QUOTES_HIST_MAXLEN": "42",
        "MAX_QUOTE_WORKERS": "7",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    settings_module = _reload_settings_module()

    for attr, key in zip(TTL_EXPORTS, overrides):
        value = getattr(settings_module, attr)
        assert value == int(overrides[key])
        assert value == getattr(settings_module.settings, attr)
