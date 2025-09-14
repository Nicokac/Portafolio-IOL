#!/usr/bin/env python3
"""Generate version.py from pyproject.toml."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
VERSION_FILE = ROOT / "version.py"

def main() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    version = data["project"]["version"]
    VERSION_FILE.write_text(f'__version__ = "{version}"\n')

if __name__ == "__main__":
    main()
