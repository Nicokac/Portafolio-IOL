"""Ensure Streamlit exit helpers are isolated to the debug module."""

from __future__ import annotations

from pathlib import Path
import re

_STOP_TOKEN = "st." + "stop"
_RERUN_TOKEN = "st." + "rerun"
_ALLOWED_PATH = Path("shared/debug/rerun_trace.py")
_PATTERNS = (
    (_STOP_TOKEN, re.compile(r"(?<![A-Za-z0-9_])" + re.escape(_STOP_TOKEN))),
    (_RERUN_TOKEN, re.compile(r"(?<![A-Za-z0-9_])" + re.escape(_RERUN_TOKEN))),
)


def test_streamlit_exit_calls_are_confined() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    allowed_file = (repo_root / _ALLOWED_PATH).resolve()
    offenders: list[tuple[Path, str]] = []

    for path in repo_root.rglob("*.py"):
        if path.is_symlink():
            continue
        resolved = path.resolve()
        if resolved == allowed_file:
            continue
        rel_path = path.relative_to(repo_root)
        if any(part.startswith(".") for part in rel_path.parts):
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for token, pattern in _PATTERNS:
            if pattern.search(content):
                offenders.append((rel_path, token))
                break

    assert not offenders, "Uso prohibido de Streamlit exit helpers:\n" + "\n".join(
        f"- {path} → {token}" for path, token in offenders
    )
