from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PREFIXES = ("fetch_", "normalize_", "calc_", "update_")
TARGET_DIRECTORIES = (
    "application",
    "controllers",
    "services",
    "shared",
    str(Path("infrastructure") / "iol"),
)


def _iter_target_files() -> list[Path]:
    files: list[Path] = []
    for directory in TARGET_DIRECTORIES:
        base = PROJECT_ROOT / directory
        if not base.exists():
            continue
        files.extend(path for path in base.rglob("*.py") if path.is_file())
    return files


def _normalize_body(node: ast.FunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def test_helper_functions_are_not_duplicated() -> None:
    duplicates: dict[str, list[tuple[str, str]]] = {}
    for path in _iter_target_files():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith(TARGET_PREFIXES):
                continue
            segment = ast.get_source_segment(source, node) or ""
            if "# noqa: Duplicated for isolation" in segment:
                continue
            body = _normalize_body(node)
            if len(body) < 4:
                continue
            canonical = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False, include_attributes=False)
            duplicates.setdefault(canonical, []).append((node.name, str(path.relative_to(PROJECT_ROOT))))

    offenders = [entries for entries in duplicates.values() if len(entries) > 1]
    assert not offenders, f"Detected duplicated helper implementations: {offenders}"


def test_compat_client_reuses_modern_implementation() -> None:
    from infrastructure.iol import client as modern_client
    from infrastructure.iol.compat import iol_client as compat_client

    assert compat_client.IOLClient is modern_client.IOLClient
