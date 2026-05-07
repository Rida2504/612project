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

# Use Python 3.12 from ~/scratch/miniconda3 (3.13 breaks many GPU wheels).
PYTHON_BIN="${PYTHON_BIN:-$HOME/scratch/miniconda3/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: $PYTHON_BIN not found. Set PYTHON_BIN env or install a Python 3.12."
    exit 1
fi
echo "== using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "== building venv at $VENV_DIR"
echo "== host: $(hostname)"

module load cuda/12.3.0/gcc/11.3.0 2>/dev/null || true

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install torch FIRST with CUDA wheels, BEFORE any other package that might
# pull torch as a dep (e.g. lpips default-installs torch 2.11 non-CUDA).
echo "== installing torch 2.5.1+cu121 FIRST so nothing upgrades it later..."
pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121

# Install gsplat before other deps (it compiles against torch)
echo "== installing gsplat..."
pip install "gsplat>=1.4" 2>&1 | tail -5 || \
    echo "!! gsplat install failed; will fall back to v2 path"

# Install lpips WITHOUT deps (its deps try to upgrade torch to non-CUDA)
echo "== installing lpips (--no-deps to preserve CUDA torch)..."
pip install --no-deps lpips

# Now the rest of requirements.txt, with torch already pinned
pip install -r "$CODE_ROOT/requirements.txt"

echo "== venv summary:"
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
python -c "import transformers; print('transformers:', transformers.__version__)" 2>/dev/null || echo "transformers: not installed"
python -c "import diffusers; print('diffusers:', diffusers.__version__)" 2>/dev/null || echo "diffusers: not installed"
python -c "import gsplat; print('gsplat: ok')" 2>/dev/null || echo "gsplat: not available (will use fallback)"
python -c "import lpips; print('lpips: ok')" 2>/dev/null || echo "lpips: not available"
python -c "import plyfile; print('plyfile: ok')"

echo "== done. Activate with: source $VENV_DIR/bin/activate"
