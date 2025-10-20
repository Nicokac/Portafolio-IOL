"""Streamlit entry point delegating to bootstrap and UI orchestration layers."""

from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path
from typing import Any

import streamlit as st

from bootstrap import TOTAL_LOAD_START as _TOTAL_LOAD_START, init_app
from bootstrap.startup import (
    is_preload_complete,
    resume_preload_worker,
    start_preload_worker,
)


def _get_startup_module():
    return importlib.import_module("bootstrap.startup")


def _ensure_ui_package_stub() -> None:
    if "ui" in sys.modules:
        return
    if any(name.startswith("ui.") for name in sys.modules):
        stub = types.ModuleType("ui")
        package_path = Path(__file__).resolve().parent / "ui"
        stub.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        sys.modules["ui"] = stub


def _get_orchestrator_module():
    _ensure_ui_package_stub()
    module = importlib.import_module("ui.orchestrator")
    module.start_preload_worker = start_preload_worker
    module.resume_preload_worker = resume_preload_worker
    module.is_preload_complete = is_preload_complete
    return module


def render_main_ui() -> None:
    _get_orchestrator_module().render_main_ui()


def _render_login_phase() -> None:
    _get_orchestrator_module()._render_login_phase()


def _render_total_load_indicator(placeholder: Any) -> None:
    _get_orchestrator_module()._render_total_load_indicator(placeholder)


def main(argv: list[str] | None = None) -> None:
    """Initialize the application and render the main UI."""

    init_app(argv)
    render_main_ui()


def __getattr__(name: str) -> Any:
    orchestrator = _get_orchestrator_module()
    if hasattr(orchestrator, name):
        return getattr(orchestrator, name)

    startup = _get_startup_module()
    if hasattr(startup, name):
        return getattr(startup, name)

    if name in {"configure_logging", "ensure_tokens_key"}:
        shared_config = importlib.import_module("shared.config")
        return getattr(shared_config, name)

    alias_targets = {
        "_lazy_attr": ("bootstrap.startup", "lazy_attr"),
        "_lazy_module": ("bootstrap.startup", "lazy_module"),
        "_PRELOAD_WORKER": ("bootstrap.startup", "_PRELOAD_WORKER"),
        "render_portfolio_section": ("controllers.portfolio.portfolio", "render_portfolio_section"),
    }
    if name in alias_targets:
        module_name, attr_name = alias_targets[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    raise AttributeError(name)


if __name__ == "__main__":
    main(sys.argv[1:])
