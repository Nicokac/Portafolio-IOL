"""Generate Python bytecode caches during deployment warm-up."""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

_TARGETS = (
    "app.py",
    "application",
    "controllers",
    "services",
    "shared",
    "ui",
)


def _compile_target(root: Path, target: str) -> bool:
    path = root / target
    if not path.exists():
        return True
    if path.is_dir():
        return bool(compileall.compile_dir(str(path), maxlevels=3, quiet=1))
    return bool(compileall.compile_file(str(path), quiet=1))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    results = [_compile_target(project_root, target) for target in _TARGETS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
