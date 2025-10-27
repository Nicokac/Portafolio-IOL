"""Tests for ensuring version metadata consistency across the project."""

from __future__ import annotations

from pathlib import Path

import tomli

from shared import version as version_module


EXPECTED_VERSION = "0.9.5.1"
EXPECTED_RELEASE_NAME = "Portafolio IOL v0.9.5.1-hotfix1"
EXPECTED_CHANGELOG_REF = ("Fase 7.5.1-hotfix1 — Streamlit Compatibility",)
EXPECTED_RELEASE_DATE = "2025-10-29"


def test_version_metadata_consistency() -> None:
    info = version_module.get_version_info()

    assert info["version"] == EXPECTED_VERSION
    assert info["codename"] == EXPECTED_RELEASE_NAME
    assert info["release_date"] == EXPECTED_RELEASE_DATE
    assert info["build_signature"] == version_module.BUILD_SIGNATURE
    assert info["changelog_ref"] == EXPECTED_CHANGELOG_REF
    assert info["stability"] == "stable"


def test_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomli.load(file)

    project_version = pyproject["project"]["version"]
    assert project_version == version_module.VERSION
    assert project_version == EXPECTED_VERSION
