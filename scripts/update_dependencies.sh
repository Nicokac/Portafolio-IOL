#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
cd "$(dirname "$0")/.."

# Packages managed from pyproject.toml
PACKAGES=(
  streamlit
  pandas
  numpy
  requests
  python-dotenv
  iolConn
  plotly
  matplotlib
  kaleido
  XlsxWriter
  tomli
  cryptography
  yfinance
  ta
)

# Upgrade packages to their latest versions
python -m pip install --upgrade "${PACKAGES[@]}"

# Run test suite to ensure updates don't break the project
pytest

# Update pinned versions in pyproject.toml based on the installed packages
python - <<'PY'
import importlib.metadata as metadata
import pathlib
import re
import tomllib

root = pathlib.Path(__file__).resolve().parents[1]
pyproject_path = root / "pyproject.toml"
data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

dependencies = data.get("project", {}).get("dependencies", [])
if not dependencies:
    raise SystemExit("No dependencies found in pyproject.toml")

updated_lines = []
for entry in dependencies:
    package, _, _ = entry.partition("==")
    if not package:
        raise SystemExit(f"Invalid dependency entry: {entry}")
    version = metadata.version(package)
    updated_lines.append(f'    "{package}=={version}",')

new_block = "dependencies = [\n" + "\n".join(updated_lines) + "\n]"

pattern = re.compile(r"dependencies = \[(?:\n.*?)*?\n\]", re.DOTALL)
original_text = pyproject_path.read_text(encoding="utf-8")
if not pattern.search(original_text):
    raise SystemExit("Could not locate dependencies block in pyproject.toml")

pyproject_path.write_text(pattern.sub(new_block, original_text, count=1), encoding="utf-8")
PY

# Regenerate requirements.txt from pyproject.toml
python scripts/sync_requirements.py
