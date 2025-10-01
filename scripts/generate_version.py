#!/usr/bin/env python3
"""Generate version.py from pyproject.toml."""
from pathlib import Path

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as _toml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
VERSION_FILE = ROOT / "version.py"

def main() -> None:
    data = _toml.loads(PYPROJECT.read_text())
    version = data["project"]["version"]
    VERSION_FILE.write_text(f'__version__ = "{version}"\n')

if __name__ == "__main__":
    main()
