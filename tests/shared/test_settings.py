"""Regression coverage for ``shared.config.Settings`` defaults and overrides."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import shared.config as config


@contextmanager
def _fresh_settings(
    monkeypatch,
    env: dict[str, str] | None = None,
    cfg_data: dict[str, Any] | None = None,
):
    """Instantiate ``Settings`` with a clean configuration context.

    ``env`` permite inyectar variables de entorno temporales y ``cfg_data``
    documenta cómo los tests simulan el contenido de ``config.json`` para
    valores como ``MIN_SCORE_THRESHOLD`` y ``MAX_RESULTS``.
    """

    original_settings = config.settings

    pre_existing_modules = {
        name: sys.modules.get(name)
        for name in ("shared.settings", "shared.cache", "application.ta_service")
    }

    shared_settings_module = pre_existing_modules["shared.settings"]
    if shared_settings_module is None:
        shared_settings_module = importlib.import_module("shared.settings")

    for key in (
        "YAHOO_FUNDAMENTALS_TTL",
        "YAHOO_QUOTES_TTL",
        "MIN_SCORE_THRESHOLD",
        "MAX_RESULTS",
    ):
        monkeypatch.delenv(key, raising=False)

    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)

    monkeypatch.setattr(config, "st", SimpleNamespace(secrets={}))
    monkeypatch.setattr(config, "_load_cfg", lambda: cfg_data or {})

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
        importlib.reload(shared_settings_module)
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


def test_settings_score_parameters_defaults(monkeypatch):
    """New scoring parameters should provide documented defaults."""

    with _fresh_settings(monkeypatch) as settings:
        assert settings.min_score_threshold == 80
        assert settings.max_results == 5


def test_settings_score_parameters_from_environment(monkeypatch):
    """Environment variables take precedence over config and defaults."""

    with _fresh_settings(
        monkeypatch,
        {
            "MIN_SCORE_THRESHOLD": "77",
            "MAX_RESULTS": "9",
        },
    ) as settings:
        assert settings.min_score_threshold == 77
        assert settings.max_results == 9


def test_settings_score_parameters_from_config(monkeypatch):
    """``config.json`` values should be honoured when env vars are absent."""

    with _fresh_settings(
        monkeypatch,
        cfg_data={"MIN_SCORE_THRESHOLD": 65, "MAX_RESULTS": 12},
    ) as settings:
        assert settings.min_score_threshold == 65
        assert settings.max_results == 12


def test_shared_settings_reload_reflects_overrides(monkeypatch):
    """``shared.settings`` should surface env overrides and keep aliases."""

    import shared.settings as shared_settings

    with _fresh_settings(
        monkeypatch,
        {
            "YAHOO_FUNDAMENTALS_TTL": "111",
            "YAHOO_QUOTES_TTL": "222",
            "MIN_SCORE_THRESHOLD": "88",
            "MAX_RESULTS": "6",
        },
    ) as settings_override:
        reloaded = importlib.reload(shared_settings)

        assert reloaded.yahoo_fundamentals_ttl == 111
        assert reloaded.yahoo_quotes_ttl == 222
        assert reloaded.min_score_threshold == 88
        assert reloaded.max_results == 6
        assert reloaded.YAHOO_FUNDAMENTALS_TTL == 111
        assert reloaded.YAHOO_QUOTES_TTL == 222
        assert reloaded.MIN_SCORE_THRESHOLD == 88
        assert reloaded.MAX_RESULTS == 6
        assert {
            "yahoo_fundamentals_ttl",
            "yahoo_quotes_ttl",
            "min_score_threshold",
            "max_results",
            "YAHOO_FUNDAMENTALS_TTL",
            "YAHOO_QUOTES_TTL",
            "MIN_SCORE_THRESHOLD",
            "MAX_RESULTS",
        } <= set(reloaded.__all__)

    restored = importlib.reload(shared_settings)
    assert restored.MIN_SCORE_THRESHOLD == config.settings.min_score_threshold
    assert restored.MAX_RESULTS == config.settings.max_results
