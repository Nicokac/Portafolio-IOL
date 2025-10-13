#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH=${1:-cache_smoke_report.json}
HOST=${SMOKE_CACHE_HOST:-127.0.0.1}
PORT=${SMOKE_CACHE_PORT:-8765}
BASE_URL=${SMOKE_CACHE_BASE_URL:-http://$HOST:$PORT}
STARTUP_TIMEOUT=${SMOKE_CACHE_STARTUP_TIMEOUT:-20}

FASTAPI_KEY=$(python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
)

IOL_KEY=$(python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
)

export FASTAPI_TOKENS_KEY="$FASTAPI_KEY"
export IOL_TOKENS_KEY="$IOL_KEY"
export CACHE_SMOKE_HOST="$HOST"
export CACHE_SMOKE_PORT="$PORT"
export CACHE_SMOKE_BASE_URL="$BASE_URL"
export CACHE_SMOKE_REPORT_PATH="$OUTPUT_PATH"
export CACHE_SMOKE_STARTUP_TIMEOUT="$STARTUP_TIMEOUT"

python - <<'PY'
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from threading import Thread

import httpx
import uvicorn

from api.main import app
from services.auth import generate_token

host = os.environ["CACHE_SMOKE_HOST"]
port = int(os.environ["CACHE_SMOKE_PORT"])
base_url = os.environ["CACHE_SMOKE_BASE_URL"]
report_path = os.environ["CACHE_SMOKE_REPORT_PATH"]
startup_timeout = float(os.environ["CACHE_SMOKE_STARTUP_TIMEOUT"])

config = uvicorn.Config(app, host=host, port=port, log_level="warning")
server = uvicorn.Server(config)

thread = Thread(target=server.run, daemon=True)
thread.start()

deadline = time.time() + startup_timeout
while not server.started:
    if time.time() > deadline:
        print(
            f"ERROR: API no disponible tras {startup_timeout}s",
            file=sys.stderr,
        )
        server.should_exit = True
        thread.join(timeout=1)
        sys.exit(1)
    time.sleep(0.1)

token = generate_token("smoke-cache", expiry=300)
headers = {"Authorization": f"Bearer {token}"}

cases = [
    {
        "name": "cache_status_unauthorized",
        "method": "GET",
        "path": "/cache/status",
        "expected_status": 401,
        "headers": {},
    },
    {
        "name": "cache_status_authorized",
        "method": "GET",
        "path": "/cache/status",
        "expected_status": 200,
        "headers": headers,
    },
    {
        "name": "cache_invalidate_invalid_pattern",
        "method": "POST",
        "path": "/cache/invalidate",
        "expected_status": 400,
        "headers": {**headers, "Content-Type": "application/json"},
        "json": {"pattern": "   "},
    },
    {
        "name": "cache_cleanup_authorized",
        "method": "POST",
        "path": "/cache/cleanup",
        "expected_status": 200,
        "headers": headers,
    },
]

results = []
latencies = []

try:
    with httpx.Client(timeout=5.0) as client:
        for case in cases:
            started = time.perf_counter()
            response = client.request(
                case["method"],
                f"{base_url}{case['path']}",
                headers=case.get("headers"),
                json=case.get("json"),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)
            ok = response.status_code == case["expected_status"]
            detail = None
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = body.get("detail")
            except Exception:
                body = response.text
            results.append(
                {
                    "name": case["name"],
                    "method": case["method"],
                    "path": case["path"],
                    "expected_status": case["expected_status"],
                    "status_code": response.status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "ok": ok,
                    "detail": detail,
                }
            )
            if not ok:
                print(
                    f"ERROR: {case['name']} esperaba {case['expected_status']} y devolvió {response.status_code}",
                    file=sys.stderr,
                )
                print(response.text, file=sys.stderr)
                raise SystemExit(1)

    average_ms = statistics.mean(latencies)
    if average_ms >= 2000:
        print(
            f"ERROR: tiempo promedio {average_ms:.2f} ms supera el umbral de 2000 ms",
            file=sys.stderr,
        )
        raise SystemExit(1)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "average_response_time_ms": round(average_ms, 2),
        "max_response_time_ms": round(max(latencies), 2),
        "min_response_time_ms": round(min(latencies), 2),
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
finally:
    server.should_exit = True
    thread.join(timeout=5)
    if thread.is_alive():
        print("ADVERTENCIA: el servidor uvicorn no se detuvo correctamente", file=sys.stderr)
        sys.exit(1)
PY
