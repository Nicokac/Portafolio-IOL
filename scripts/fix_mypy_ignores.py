"""Normalize ``type: ignore`` annotations based on the latest mypy output."""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import subprocess
from collections.abc import Iterable, Sequence

LOGGER = logging.getLogger(__name__)

ERR_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+): error: .+ \[(?P<code>[-a-z0-9]+)\]$")
UNUSED_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+): error: unused 'type: ignore' comment")

DEFAULT_MYPY_ARGS = (
    "mypy",
    "--ignore-missing-imports",
    "--warn-unused-ignores",
    "--show-error-codes",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta mypy y normaliza anotaciones `type: ignore`.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=(".",),
        help=("Rutas o módulos a validar con mypy (por defecto el repositorio completo)."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra los cambios detectados sin modificar archivos.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run_mypy(paths: Iterable[str]) -> list[str]:
    """Run mypy and return its stdout output line by line."""

    command = [*DEFAULT_MYPY_ARGS, *paths]
    LOGGER.debug("Executing command: %s", " ".join(command))
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stderr.strip():
        LOGGER.debug("mypy stderr:\n%s", proc.stderr.strip())
    return proc.stdout.splitlines()


def _group_mypy_findings(
    lines: Iterable[str],
) -> tuple[dict[tuple[str, int], set[str]], set[tuple[str, int]]]:
    """Extract error codes and unused ignore locations from mypy output."""

    by_line_codes: dict[tuple[str, int], set[str]] = {}
    unused_ignores: set[tuple[str, int]] = set()

    for line in lines:
        err_match = ERR_RE.match(line)
        unused_match = UNUSED_RE.match(line)
        if err_match:
            key = (err_match.group("path"), int(err_match.group("line")))
            by_line_codes.setdefault(key, set()).add(err_match.group("code"))
        elif unused_match:
            key = (unused_match.group("path"), int(unused_match.group("line")))
            unused_ignores.add(key)

    return by_line_codes, unused_ignores


def _apply_error_codes(
    annotations: dict[tuple[str, int], set[str]],
    *,
    dry_run: bool,
) -> int:
    """Add the discovered error codes to existing ``type: ignore`` comments."""

    updated = 0
    for (path, line_no), codes in sorted(annotations.items()):
        file_path = pathlib.Path(path)
        if not file_path.exists():
            LOGGER.debug("Skipping missing path: %s", path)
            continue
        source_lines = file_path.read_text(encoding="utf-8").splitlines()
        idx = line_no - 1
        if not 0 <= idx < len(source_lines):
            LOGGER.debug("Line %s:%s fuera de rango", path, line_no)
            continue
        line = source_lines[idx]
        if "# type: ignore" not in line or "[" in line:
            continue
        replacement = line.replace(
            "# type: ignore",
            f"# type: ignore[{','.join(sorted(codes))}]",
        )
        LOGGER.info("Actualizando %s:%s -> %s", path, line_no, replacement.strip())
        if not dry_run:
            source_lines[idx] = replacement
            file_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
        updated += 1
    return updated


def _remove_unused_ignores(
    unused: Iterable[tuple[str, int]],
    *,
    dry_run: bool,
) -> int:
    """Remove redundant ``type: ignore`` annotations."""

    removed = 0
    for path, line_no in sorted(unused):
        file_path = pathlib.Path(path)
        if not file_path.exists():
            LOGGER.debug("Skipping missing path: %s", path)
            continue
        source_lines = file_path.read_text(encoding="utf-8").splitlines()
        idx = line_no - 1
        if not 0 <= idx < len(source_lines):
            LOGGER.debug("Line %s:%s fuera de rango", path, line_no)
            continue
        line = source_lines[idx]
        if "# type: ignore" not in line:
            continue
        LOGGER.info("Eliminando anotación innecesaria en %s:%s", path, line_no)
        if not dry_run:
            source_lines[idx] = line.replace("# type: ignore", "").rstrip()
            file_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
        removed += 1
    return removed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    stdout_lines = run_mypy(args.paths)
    annotations, unused = _group_mypy_findings(stdout_lines)

    updated = _apply_error_codes(annotations, dry_run=args.dry_run)
    removed = _remove_unused_ignores(unused, dry_run=args.dry_run)

    LOGGER.info(
        "Resumido: %s anotaciones actualizadas, %s ignoradas eliminadas",
        updated,
        removed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
