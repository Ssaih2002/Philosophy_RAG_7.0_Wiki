#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/4] Checking Python..."
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "python3 not found. Please install Python 3.11+."
  exit 1
fi

echo "[2/4] Creating venv if needed..."
if [ ! -x ".venv/bin/python" ]; then
  "$PY" -m venv .venv
fi

echo "[3/4] Installing dependencies..."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt
".venv/bin/python" -m pip install uvicorn

echo "[4/4] Starting backend and opening frontend..."
if [ -f "tools/ensure_torch_accel.py" ]; then
  # On macOS this script will no-op (uses MPS if available)
  ".venv/bin/python" "tools/ensure_torch_accel.py" || true
fi

open "frontend.html" || true

echo "Backend starting at http://127.0.0.1:8000"
exec ".venv/bin/python" -m uvicorn web_app:app --host 127.0.0.1 --port 8000

