import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_build_environment_snapshot_gathers_runtime_details(monkeypatch, caplog):
    module = importlib.reload(importlib.import_module("services.environment_snapshot"))

    fake_time = SimpleNamespace(time=lambda: 1_234_567_890.0)
    monkeypatch.setattr(module, "time", fake_time)
    monkeypatch.setattr(module, "TimeProvider", SimpleNamespace(now=lambda: "2024-05-18 10:00:00"))

    monkeypatch.setattr(
        module,
        "platform",
        SimpleNamespace(
            python_implementation=lambda: "CPython",
            system=lambda: "Linux",
            release=lambda: "6.2.0",
            machine=lambda: "x86_64",
        ),
    )
    monkeypatch.setattr(module, "sys", SimpleNamespace(version="3.11.3", executable="/usr/bin/python"))

    env = {
        "STREAMLIT_ENV": "ci",
        "CACHE_TTL_QUOTES": "30",
        "IOL_PASSWORD": "super-secret",
        "EMPTY_VALUE": " ",
    }
    packages = [
        {"name": "zeta", "version": "2"},
        {"name": "alpha", "version": "1"},
    ]

    with caplog.at_level(logging.INFO, logger=module.logger.name):
        snapshot = module.build_environment_snapshot(
            env=env,
            packages=packages,
            include_installed_packages=False,
        )

    assert snapshot["event"] == "environment.snapshot"
    assert snapshot["timestamp"] == "2024-05-18 10:00:00"
    assert snapshot["ts"] == 1_234_567_890.0

    runtime = snapshot["runtime"]
    assert runtime["python"]["version"] == "3.11.3"
    assert runtime["python"]["implementation"] == "CPython"
    assert runtime["platform"] == {
        "system": "Linux",
        "release": "6.2.0",
        "machine": "x86_64",
    }
    assert runtime["executable"] == "/usr/bin/python"

    environment = snapshot["environment"]
    assert environment["CACHE_TTL_QUOTES"] == "30"
    assert environment["STREAMLIT_ENV"] == "ci"
    assert environment["IOL_PASSWORD"] == "***"
    assert "EMPTY_VALUE" not in environment

    packages_snapshot = snapshot["packages"]
    assert packages_snapshot == [
        {"name": "alpha", "version": "1"},
        {"name": "zeta", "version": "2"},
    ]

    record = next(rec for rec in caplog.records if rec.message == "environment.snapshot")
    assert record.analysis == snapshot
