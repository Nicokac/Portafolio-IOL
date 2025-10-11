"""Sesiones de QA locales para Portafolio IOL."""

from __future__ import annotations

from pathlib import Path

import nox

PROJECT_DIR = Path(__file__).parent
SOURCE_DIRS = ("application", "controllers", "services", "shared", "tests")

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ("lint", "typecheck", "tests", "security")


def _install_requirements(session: nox.Session, *extra: str) -> None:
    """Instala los requisitos necesarios para la sesión."""

    requirements = PROJECT_DIR / "requirements-dev.txt"
    if requirements.exists():
        session.install("-r", str(requirements))
    else:
        session.log("requirements-dev.txt no encontrado, continuando sin instalar dev deps")

    if extra:
        session.install(*extra)


@nox.session
def lint(session: nox.Session) -> None:
    """Ejecuta flake8 sobre los módulos principales."""

    _install_requirements(session, "flake8>=7.0.0")
    session.run("flake8", *SOURCE_DIRS)


@nox.session
def typecheck(session: nox.Session) -> None:
    """Valida los tipos usando mypy."""

    _install_requirements(session, "mypy>=1.11.0")
    session.run("mypy", *SOURCE_DIRS)


@nox.session
def tests(session: nox.Session) -> None:
    """Ejecuta la suite de pytest con cobertura."""

    _install_requirements(session, "pytest>=8.0.0", "pytest-cov>=4.1.0")
    session.install("-r", "requirements.txt")
    session.run("pytest", "--cov-report=term-missing")


@nox.session
def security(session: nox.Session) -> None:
    """Ejecuta verificaciones de seguridad con bandit y pip-audit."""

    _install_requirements(session, "bandit>=1.7.9", "pip-audit>=2.7.3")
    session.run("bandit", "-q", "-r", *SOURCE_DIRS)
    session.run("pip-audit")
