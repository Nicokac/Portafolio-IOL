"""Entry point for ``python -m tools`` exposing bundled CLI utilities."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from . import __version__, benchmark_report

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools",
        description="Herramientas auxiliares para Portafolio IOL",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Comando a ejecutar (por ejemplo benchmark-report)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Argumentos adicionales para el comando seleccionado",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Portafolio IOL tools {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    command = parsed.command or ""
    args = parsed.args or []

    if command in {"benchmark-report", "benchmark_report"}:
        LOGGER.debug("Delegando en tools.benchmark_report con args: %s", args)
        return benchmark_report.main(args)

    if not command:
        parser.print_help()
        return 0

    parser.error(f"Comando desconocido: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
