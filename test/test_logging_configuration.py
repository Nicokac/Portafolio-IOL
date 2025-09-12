import importlib
import logging


def _load_config():
    import shared.config as config
    importlib.reload(config)
    return config


def _reset_logging():
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)


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
