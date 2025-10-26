from __future__ import annotations

import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iter_python_files(base: Path) -> list[Path]:
    return [path for path in base.rglob("*.py") if path.is_file()]


def _extract_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names if alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_release_version_matches_hotfix() -> None:
    from shared import version as version_module

    assert version_module.__version__ == "0.9.0.1"

    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(?P<version>[^"]+)"', pyproject_text, flags=re.MULTILINE)
    assert match, "pyproject.toml must declare a project.version"
    assert match.group("version") == version_module.__version__


def test_shared_layer_avoids_upper_imports() -> None:
    disallowed = ("controllers", "application", "services")
    offenders: list[tuple[str, str]] = []
    for file_path in _iter_python_files(PROJECT_ROOT / "shared"):
        for module in _extract_imports(file_path):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in disallowed):
                offenders.append((str(file_path.relative_to(PROJECT_ROOT)), module))

    assert not offenders, f"shared layer must not depend on upper layers: {offenders}"
