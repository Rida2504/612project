#!/bin/bash
# build_dgr_cu121.sh — build diff-gaussian-rasterization for cu121 main venv.
#
# MUST run on a compute node (H100, nvcc available).

set -uo pipefail

LOG="$HOME/scratch/phase4/textworld-vr/logs/p3_build_dgr_cu121.log"
mkdir -p "$(dirname "$LOG")"
exec > "$LOG" 2>&1

echo "== [$(date -Is)] P3 starting on $(hostname)"

module load cuda/12.3.0/gcc/11.3.0 2>/dev/null || true
source "$HOME/scratch/phase4/textworld-vr/shared/venv/bin/activate"
python --version
which nvcc && nvcc --version | tail -2

export TORCH_CUDA_ARCH_LIST="9.0"  # H100
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.3}"

SUBMOD="$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D/submodules/diff-gaussian-rasterization"
[ -d "$SUBMOD" ] || { echo "ERROR: submodule dir missing at $SUBMOD"; exit 1; }

echo "== building from $SUBMOD"
cd "$SUBMOD"
pip install --no-cache-dir --no-build-isolation . 2>&1 | tail -30

echo "== import smoke"
python - <<'PY'
import diff_gaussian_rasterization as d
print("ok:", d.__file__)
from diff_gaussian_rasterization import _C
print("cuda ext ok:", _C.__file__ if hasattr(_C, "__file__") else "bound")
PY

echo "== [$(date -Is)] P3 finished"
echo "== DONE =="
