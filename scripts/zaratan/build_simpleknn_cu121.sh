#!/bin/bash
# build_simpleknn_cu121.sh — build simple-knn for cu121 main venv.
#
# MUST run on a compute node (H100, nvcc available).

set -uo pipefail

LOG="$HOME/scratch/phase4/textworld-vr/logs/p4_build_simpleknn_cu121.log"
mkdir -p "$(dirname "$LOG")"
exec > "$LOG" 2>&1

echo "== [$(date -Is)] P4 starting on $(hostname)"

module load cuda/12.3.0/gcc/11.3.0 2>/dev/null || true
source "$HOME/scratch/phase4/textworld-vr/shared/venv/bin/activate"
python --version
which nvcc && nvcc --version | tail -2

export TORCH_CUDA_ARCH_LIST="9.0"  # H100
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.3}"

SUBMOD="$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D/submodules/simple-knn"
[ -d "$SUBMOD" ] || { echo "ERROR: submodule dir missing at $SUBMOD"; exit 1; }

echo "== building from $SUBMOD"
cd "$SUBMOD"
pip install --no-cache-dir --no-build-isolation . 2>&1 | tail -30

echo "== import smoke"
python - <<'PY'
from simple_knn._C import distCUDA2
import torch
pts = torch.rand(100, 3).cuda()
d = distCUDA2(pts)
print("ok distCUDA2:", d.shape, d.dtype)
PY

echo "== [$(date -Is)] P4 finished"
echo "== DONE =="
