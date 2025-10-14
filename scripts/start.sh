#!/bin/sh
set -e

export UV_USE_REQUIREMENTS=true
export PYTHONDONTWRITEBYTECODE=0

if [ "${ENABLE_BYTECODE_WARMUP:-1}" != "0" ]; then
  python scripts/warmup_bytecode.py || echo "Advertencia: warm-up de bytecode falló" >&2
fi

if [ -z "$IOL_TOKENS_KEY" ]; then
  echo "Error: IOL_TOKENS_KEY debe estar definida" >&2
  exit 1
fi

exec streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0 "$@"
