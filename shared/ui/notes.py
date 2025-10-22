"""Helpers for formatting backend notes in UI components."""

from __future__ import annotations

from typing import Mapping, Sequence

NOTE_SEVERITIES: Mapping[str, Mapping[str, Sequence[str]]] = {
    "warning": {
        "prefixes": ("⚠️",),
        "keywords": (
            "recortaron los resultados",
            "mejores",
            "máximo solicitado",
            "maximo solicitado",
            "no se encontraron candidatos",
            "sin candidatos",
            "solo se encontraron",
            "solo se encontro",
            "solo encontramos",
            "mínimo esperado",
            "minimo esperado",
            "datos simulados",
        ),
    },
    "info": {
        "prefixes": ("ℹ️",),
    },
    "success": {
        "prefixes": ("✅", "✔️"),
    },
    "error": {
        "prefixes": ("❌", "🚫"),
    },
}

NOTE_RENDERING: Mapping[str, Mapping[str, object]] = {
    "warning": {"icon": ":warning:", "emphasize": True},
    "info": {"icon": ":information_source:", "emphasize": False},
    "success": {"icon": ":white_check_mark:", "emphasize": True},
    "error": {"icon": ":x:", "emphasize": True},
}


def classify_note(note: str) -> tuple[str | None, str, bool]:
    """Return the severity category for a note and its normalized content."""

    stripped = note.strip()
    if not stripped:
        return None, "", False

    severity = None
    content = stripped
    matched_by_prefix = False

    for candidate, metadata in NOTE_SEVERITIES.items():
        prefixes = metadata.get("prefixes")
        if not prefixes:
            continue
        for prefix in prefixes:
            if content.startswith(prefix):
                matched_by_prefix = True
                severity = candidate
                content = content[len(prefix) :].lstrip()
                break
        if matched_by_prefix:
            break

    normalized = content.casefold()
    if severity is None and normalized:
        for candidate, metadata in NOTE_SEVERITIES.items():
            keywords = metadata.get("keywords")
            if not keywords:
                continue
            for keyword in keywords:
                if keyword in normalized:
                    severity = candidate
                    break
            if severity is not None:
                break

    return severity, content, matched_by_prefix


def format_note(note: str) -> str:
    """Format backend notes based on their severity category."""

    severity, content, matched_by_prefix = classify_note(note)
    if severity is None:
        return note.strip()

    rendering = NOTE_RENDERING.get(severity, {"icon": "", "emphasize": False})
    icon = rendering.get("icon", "") or ""
    emphasize = bool(rendering.get("emphasize"))

    if not content:
        stripped = note.strip()
        return icon if matched_by_prefix else stripped

    formatted = f"**{content}**" if emphasize else content

    if matched_by_prefix and icon:
        return f"{icon} {formatted}" if content else icon

    if matched_by_prefix:
        return formatted

    return formatted
