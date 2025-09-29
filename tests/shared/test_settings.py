"""Regression coverage for ``shared.config.Settings`` TTL defaults and overrides."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import shared.config as config


@contextmanager
def _fresh_settings(monkeypatch, env: dict[str, str] | None = None):
    """Instantiate ``Settings`` with a clean configuration context."""

    original_settings = config.settings

    pre_existing_modules = {
        name: sys.modules.get(name)
        for name in ("shared.settings", "shared.cache", "application.ta_service")
    }

    shared_settings_module = pre_existing_modules["shared.settings"]
    if shared_settings_module is None:
        shared_settings_module = importlib.import_module("shared.settings")

    for key in ("YAHOO_FUNDAMENTALS_TTL", "YAHOO_QUOTES_TTL"):
        monkeypatch.delenv(key, raising=False)

    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)

    importlib.reload(config)
    monkeypatch.setattr(config, "st", SimpleNamespace(secrets={}))
    monkeypatch.setattr(config, "_load_cfg", lambda: {})

    fresh_settings = config.Settings()
    config.settings = fresh_settings

    shared_settings_module = importlib.reload(shared_settings_module)

    reloaded_modules: dict[str, ModuleType] = {}
    for name in ("shared.cache", "application.ta_service"):
        module = pre_existing_modules[name]
        if module is not None:
            reloaded_modules[name] = importlib.reload(module)

    try:
        yield fresh_settings
    finally:
        config.settings = original_settings
        shared_settings_module = importlib.reload(shared_settings_module)

        for name in reloaded_modules:
            module = sys.modules.get(name)
            if module is not None:
                importlib.reload(module)


def test_settings_use_expected_default_ttls(monkeypatch):
    """Defaults should fall back to 3600s (fundamentals) and 300s (quotes)."""

    with _fresh_settings(monkeypatch) as settings:
        assert settings.YAHOO_FUNDAMENTALS_TTL == 3600
        assert settings.YAHOO_QUOTES_TTL == 300


def test_settings_respect_environment_overrides(monkeypatch):
    """Environment variables should override the default TTL values."""

    with _fresh_settings(
        monkeypatch,
        {
            "YAHOO_FUNDAMENTALS_TTL": "42",
            "YAHOO_QUOTES_TTL": "99",
        },
    ) as settings:
        assert settings.YAHOO_FUNDAMENTALS_TTL == 42
        assert settings.YAHOO_QUOTES_TTL == 99
