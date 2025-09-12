import importlib
import json
import logging

import pytest

def _load_config():
    import shared.config as config
    importlib.reload(config)
    return config


def _reset_logging():
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)

@pytest.mark.parametrize(
    "env_level, env_format, expected_level, is_json",
    [
        (None, None, logging.INFO, False),
        ("DEBUG", "plain", logging.DEBUG, False),
        ("DEBUG", "json", logging.DEBUG, True),
        ("ERROR", "plain", logging.ERROR, False),
        ("ERROR", "json", logging.ERROR, True),
    ],
)
def test_configure_logging_combinations(
    env_level,
    env_format,
    expected_level,
    is_json,
    monkeypatch,
    caplog,
):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    if env_level:
        monkeypatch.setenv("LOG_LEVEL", env_level)
    if env_format:
        monkeypatch.setenv("LOG_FORMAT", env_format)

    config = _load_config()
    _reset_logging()
    config.configure_logging()

    root = logging.getLogger()
    root.addHandler(caplog.handler)
    with caplog.at_level(expected_level):
        root.log(expected_level, "hello")
    root.removeHandler(caplog.handler)

    assert root.level == expected_level
    handler = root.handlers[0]
    assert any(
        r.levelno == expected_level and r.message == "hello"
        for r in caplog.records
    )
    record = caplog.records[0]
    formatted = handler.format(record)

    if is_json:
        assert isinstance(handler.formatter, config.JsonFormatter)
        data = json.loads(formatted)
        assert data["level"] == logging.getLevelName(expected_level)
        assert data["message"] == "hello"
    else:
        assert not isinstance(handler.formatter, config.JsonFormatter)
        assert logging.getLevelName(expected_level) in formatted
        assert "hello" in formatted


def test_configure_logging_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "json")
    config = _load_config()
    _reset_logging()

    config.configure_logging()

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert isinstance(root.handlers[0].formatter, config.JsonFormatter)


def test_configure_logging_args_override(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    config = _load_config()
    _reset_logging()

    config.configure_logging(level="WARNING", json_format=True)

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert isinstance(root.handlers[0].formatter, config.JsonFormatter)
