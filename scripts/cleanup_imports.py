"""Utility script to harmonize formatting and imports across the project.

Parte de la capa scripts. No ejecutar código en import.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    """Run Ruff lint fixes and formatting passes."""

    print("🧹 Running Ruff + isort cleanup...")
    result_fix = subprocess.run(["ruff", "check", ".", "--fix"], check=False)
    result_format = subprocess.run(["ruff", "format", "."], check=False)
    if result_fix.returncode or result_format.returncode:
        print("⚠️ Cleanup completed with warnings.")
    else:
        print("✅ Cleanup completed.")
    return 0 if not (result_fix.returncode or result_format.returncode) else 1


if __name__ == "__main__":
    sys.exit(main())
