"""Integration checks ensuring the cleanup tasks remain in place."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

FORBIDDEN_SEGMENTS = ("deprecated", "legacy", "mock")
REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_SCAN_DIRS = {
    ".git",
    "archive",
    "logs",
    "mypy_report",
    "tokens",
    "__pycache__",
}
SKIP_IMPORT_PARTS = {
    "tests",
    "__pycache__",
    "archive",
    "logs",
    "mypy_report",
    "tokens",
    "docs",
    "banners",
    "data",
}
SCANNED_PATHS = [
    path
    for path in REPO_ROOT.rglob("*")
    if path.is_file() and not any(part in SKIP_SCAN_DIRS for part in path.parts)
]
PACKAGE_ROOTS = {
    path
    for path in REPO_ROOT.iterdir()
    if path.is_dir() and (path / "__init__.py").exists()
}
SKIP_IMPORT_MODULES = {"noxfile"}
ROOT_LEVEL_MODULES = {
    path.stem
    for path in REPO_ROOT.glob("*.py")
    if path.is_file()
    and not path.name.startswith(".")
    and path.stem not in SKIP_IMPORT_MODULES
}


def test_no_forbidden_file_names() -> None:
    """Assert there are no files left with cleanup keywords in their name."""

    violations: list[str] = []
    for path in SCANNED_PATHS:
        lower_name = path.name.lower()
        segments = [segment for segment in filter(None, _split_name(lower_name))]

        if any(token in lower_name for token in FORBIDDEN_SEGMENTS):
            violations.append(str(path))
        elif "old" in segments:
            violations.append(str(path))

    assert not violations, "Forbidden file names detected:\n" + "\n".join(sorted(violations))


def _split_name(name: str) -> Iterable[str]:
    """Split the file name into alphanumeric chunks."""

    chunk = []
    for char in name:
        if char.isalnum():
            chunk.append(char)
            continue
        if chunk:
            yield "".join(chunk)
            chunk = []
    if chunk:
        yield "".join(chunk)


def test_all_modules_importable() -> None:
    """Import all package modules to ensure consolidation integrity."""

    sys.path.insert(0, str(REPO_ROOT))
    failures: list[str] = []

    for module_name in sorted(_collect_module_names()):
        if module_name in SKIP_IMPORT_MODULES:
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - reported via assertion
            failures.append(f"{module_name}: {exc}")

    assert not failures, "Modules failed to import:\n" + "\n".join(failures)


def _collect_module_names() -> set[str]:
    """Return the full dotted path for each importable module in scope."""

    module_names: set[str] = set(ROOT_LEVEL_MODULES)
    for package_path in PACKAGE_ROOTS:
        for file_path in package_path.rglob("*.py"):
            relative = file_path.relative_to(REPO_ROOT)
            if relative.name == "__init__.py":
                relative_parts = relative.parts[:-1]
            else:
                relative_parts = relative.with_suffix("").parts
            if any(part in SKIP_IMPORT_PARTS for part in relative_parts):
                continue
            module_name = ".".join(relative_parts)
            if module_name:
                module_names.add(module_name)
    return module_names


def test_version_bumped_to_0_9_0() -> None:
    """Ensure the shared version module exposes the expected release."""

    from shared.version import __version__

    assert __version__ == "0.9.0"
