#!/bin/bash
# run.sh — invoked by systemd. Stages models (idempotent, fast on warm cache)
# then exec uvicorn under the main conda env.

set -euo pipefail

REPO="${REPO:-/mnt/src/textworld-vr}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
HF_HOME="${HF_HOME:-/mnt/models/hf_cache}"
LP3D_ROOT="${LP3D_ROOT:-/opt/LayerPano3D}"
LP3D_CHECKPOINTS="${LP3D_CHECKPOINTS:-$LP3D_ROOT/checkpoints}"

# Idempotent re-stage; on warm cache this is a no-op (~1s).
"$CONDA_DIR/envs/main/bin/python" "$REPO/deploy/server/stage_models.py" \
    --hf-cache "$HF_HOME" --checkpoints "$LP3D_CHECKPOINTS"

cd "$REPO"
# Run uvicorn from inside the deploy/ dir so `server.api:app` resolves the
# same way it does inside the docker image (where /app/server is the package).
cd "$REPO/deploy"
exec "$CONDA_DIR/envs/main/bin/uvicorn" server.api:app \
    --host 0.0.0.0 --port 8000 --workers 1 --log-level info
