#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
cd "$(dirname "$0")/.."

# Packages managed in requirements.txt
PACKAGES=(
  streamlit
  pandas
  numpy
  requests
  python-dotenv
  iolConn
  plotly
  kaleido
  cryptography
  yfinance
  ta
)

# Upgrade packages to their latest versions
python -m pip install --upgrade "${PACKAGES[@]}"

# Run test suite to ensure updates don't break the project
pytest

# Freeze the exact versions back into requirements.txt
python -m pip freeze | grep -i -E '^(streamlit|pandas|numpy|requests|python-dotenv|iolConn|plotly|kaleido|cryptography|yfinance|ta)==' > requirements.txt
