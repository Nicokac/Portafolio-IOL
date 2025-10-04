#!/usr/bin/env python3
"""Synchronize requirements.txt with pyproject.toml dependencies."""

from __future__ import annotations

import pathlib
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for <=3.10
    import tomli as tomllib  # type: ignore


HEADER = """# This file is auto-generated from pyproject.toml.
# Run `python scripts/sync_requirements.py` after editing [project.dependencies]."""


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    requirements_path = repo_root / "requirements.txt"

    with pyproject_path.open("rb") as stream:
        data = tomllib.load(stream)

    dependencies = data.get("project", {}).get("dependencies", [])
    if not dependencies:
        print("No dependencies found in pyproject.toml", file=sys.stderr)
        return 1

    content_lines = [HEADER, *dependencies]
    requirements_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
