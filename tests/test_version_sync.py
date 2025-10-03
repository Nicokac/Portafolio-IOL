from __future__ import annotations

from pathlib import Path
import sys

try:  # pragma: no cover - Python < 3.11 fallback mirrors runtime module
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml  # type: ignore[import-untyped]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared import version


def test_default_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project_version = _toml.loads(pyproject_path.read_text(encoding="utf-8"))["project"][
        "version"
    ]

    assert version.DEFAULT_VERSION == project_version == version.__version__
