#!/bin/bash
# build_venv.sh — create a shared Python venv at ~/scratch/.../shared/venv/
#
# Run ONCE per cluster-workspace lifetime (or when requirements change).
# Must be run on a GPU node (for CUDA wheels like torch, gsplat).
#
# The venv lives on BeeGFS so both nodes can activate it. First activation per
# node is slow; subsequent are cached by the kernel page cache.

set -euo pipefail

SCRATCH_ROOT="$HOME/scratch/phase4/textworld-vr"
VENV_DIR="$SCRATCH_ROOT/shared/venv"
CODE_ROOT="$SCRATCH_ROOT/code"

echo "== building venv at $VENV_DIR"
echo "== host: $(hostname)"

module load cuda/12.3.0/gcc/11.3.0 2>/dev/null || true

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install from requirements.txt
pip install -r "$CODE_ROOT/requirements.txt"

# Install CUDA-only extras (these may take a while)
echo "== installing CUDA extras..."
pip install 'torch>=2.5,<2.9' --index-url https://download.pytorch.org/whl/cu121 || \
    pip install 'torch>=2.5,<2.9'
pip install 'gsplat>=1.4' 2>&1 | tail -5 || \
    echo "gsplat install failed; will fall back to v2-fallback"

echo "== venv summary:"
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
python -c "import transformers; print('transformers:', transformers.__version__)" 2>/dev/null || echo "transformers: not installed"
python -c "import diffusers; print('diffusers:', diffusers.__version__)" 2>/dev/null || echo "diffusers: not installed"
python -c "import gsplat; print('gsplat: ok')" 2>/dev/null || echo "gsplat: not available (will use fallback)"
python -c "import lpips; print('lpips: ok')" 2>/dev/null || echo "lpips: not available"
python -c "import plyfile; print('plyfile: ok')"

echo "== done. Activate with: source $VENV_DIR/bin/activate"
