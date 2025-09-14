#!/bin/sh
set -e

if [ -z "$IOL_TOKENS_KEY" ]; then
  echo "Error: IOL_TOKENS_KEY debe estar definida" >&2
  exit 1
fi

exec streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0 "$@"
