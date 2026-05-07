#!/bin/bash
# entrypoint.sh — orchestrate Ray Serve + nginx for the TextWorld VR image.
set -euo pipefail

echo "[$(date -Is)] boot: TextWorld VR deploy image"
echo "   HF_HOME=$HF_HOME  SCENES_DIR=$SCENES_DIR  LP3D_ROOT=$LP3D_ROOT"
echo "   GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

# 1. Stage LP3D checkpoints from HF on first start if missing (idempotent).
mkdir -p "$LP3D_CHECKPOINTS" "$SCENES_DIR" "$HF_HOME"
/opt/conda/envs/main/bin/python /app/server/stage_models.py \
    --hf-cache "$HF_HOME" --checkpoints "$LP3D_CHECKPOINTS" \
    || { echo "model staging failed"; exit 1; }

# 2. Start nginx (static viewer + splat file server) in background.
nginx -g 'daemon off;' &
NGINX_PID=$!
echo "[entrypoint] nginx pid=$NGINX_PID"

# 3. Start the FastAPI app under uvicorn.
#    --workers 1 is required: one process owns the GPU, the asyncio.Semaphore(1)
#    that serializes jobs, and the in-memory _JOBS dict. Multiple workers would
#    each load SDXL onto the same GPU and fight for it.
export PYTHONPATH=/app:${PYTHONPATH:-}
cd /app
exec /opt/conda/envs/main/bin/uvicorn server.api:app \
    --host 0.0.0.0 --port 8000 --workers 1 --log-level info
