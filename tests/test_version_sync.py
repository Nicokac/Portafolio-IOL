from __future__ import annotations

import sys
from pathlib import Path

from shared import version

try:  # pragma: no cover - Python < 3.11 fallback mirrors runtime module
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml  # type: ignore[import-untyped]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_default_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project_version = _toml.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["version"]

    assert version.DEFAULT_VERSION == project_version == version.__version__
