#!/usr/bin/env bash
# VocalTrace — one-shot environment bootstrap (macOS / Linux).
#
# Creates a virtual environment, installs core dependencies, and runs a
# smoke test. Idempotent: safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "== VocalTrace setup =="

# --- 0. Python version ------------------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found. Install Python 3.9+ first." >&2
    exit 1
fi
"$PYTHON" - <<'EOF'
import sys
if sys.version_info < (3, 9):
    raise SystemExit(f"error: Python 3.9+ required, found {sys.version.split()[0]}")
print(f"Python {sys.version.split()[0]} OK")
EOF

# --- 1. Virtual environment -------------------------------------------------
if [ ! -x .venv/bin/python ]; then
    echo "-- creating virtual environment in .venv"
    "$PYTHON" -m venv .venv
else
    echo "-- reusing existing .venv"
fi
PY=".venv/bin/python"

# --- 2. Dependencies --------------------------------------------------------
echo "-- installing dependencies (this can take a few minutes)"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r requirements.txt

# --- 3. Platform-specific niceties (non-fatal) ------------------------------
case "$(uname -s)" in
    Darwin)
        if ! "$PY" -c "import xgboost" >/dev/null 2>&1; then
            echo "-- xgboost failed to import; on macOS this usually means libomp is missing:"
            echo "     brew install libomp"
        fi
        ;;
esac

# --- 4. Smoke test ----------------------------------------------------------
echo "-- running smoke test"
"$PY" - <<'EOF'
import numpy as np
import vocaltrace
from vocaltrace.features import FeatureExtractor

fe = FeatureExtractor()
audio = np.random.default_rng(42).normal(0, 0.1, 16000).astype(np.float32)
features = fe.extract_all(audio)
vector = fe.to_flat_vector(features)
assert vector.size == fe.flat_dim
print(f"VocalTrace {vocaltrace.__version__} ready (flat vector dim={vector.size})")
EOF

echo ""
echo "Setup complete. Try:"
echo "  .venv/bin/python main.py detect --file data/fake/sample1.wav"
echo "  .venv/bin/python -m pytest tests/ -q"
