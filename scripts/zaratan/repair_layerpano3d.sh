#!/bin/bash
# repair_layerpano3d.sh — second pass fixing what install_layerpano3d.sh couldn't.
#
# Known breakage from round 1:
#  - natten failed (shi-labs.com SSL cert expired). Skip, LP3D's top-level code
#    doesn't import natten directly; it's a transitive dep of some sub-package.
#  - pip install -r requirements.txt bailed early → diffusers never installed.
#  - diff-gaussian-rasterization failed because pip's build-isolation env has
#    no torch. Fix with --no-build-isolation.
#  - simple-knn: same build-isolation issue.
#  - 360monodepth cmake failed. This one DOES need serious deps. We'll replace
#    its depth step with our DA-v2 pipeline at inference time instead.

set -uo pipefail

LOG="$HOME/scratch/phase4/textworld-vr/logs/lp3d_repair.log"
CONDA="$HOME/scratch/miniconda3"
REPO="$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D"
exec > "$LOG" 2>&1

echo "== [$(date -Is)] lp3d repair starting on $(hostname)"

source "$CONDA/etc/profile.d/conda.sh"
conda activate lp3d
python --version
pip --version

# 1. Install LP3D python deps individually (skip natten, skip torch-related since done)
echo "== installing LP3D runtime python deps"
pip install --no-cache-dir --timeout 30 \
    diffusers==0.32.0 \
    transformers==4.45.2 \
    accelerate \
    safetensors \
    sentencepiece==0.2.0 \
    peft==0.14.0 \
    kornia==0.8.0 \
    einops==0.4.1 \
    plyfile==1.1 \
    opencv-python \
    scikit-image==0.24.0 \
    pytorch-lightning==2.4.0 \
    pyyaml \
    tqdm \
    omegaconf \
    loguru==0.7.3 \
    scikit-learn==1.6.1 \
    easydict==1.9.0 \
    pandas==2.2.3 \
    matplotlib \
    open_clip_torch==2.30.0 \
    hydra-core==1.1.0 \
    albumentations==0.5.2 \
    timm==0.4.12 --no-deps --no-build-isolation 2>&1 | tail -8 || true

# 2. diff-gaussian-rasterization with --no-build-isolation (so torch is visible)
echo "== installing diff-gaussian-rasterization --no-build-isolation"
cd "$REPO"
module load cuda/12.3.0/gcc/11.3.0 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="9.0"   # H100
pip install --no-cache-dir --no-build-isolation -e submodules/diff-gaussian-rasterization 2>&1 | tail -10 || \
    echo "!! diff-gaussian-rasterization still failed"

# 3. simple-knn with --no-build-isolation
echo "== installing simple-knn --no-build-isolation"
pip install --no-cache-dir --no-build-isolation -e submodules/simple-knn 2>&1 | tail -10 || \
    echo "!! simple-knn still failed"

# 4. Import smoke
echo "== import smoke (final)"
python - <<PY 2>&1 | tail -30
import importlib, sys
for m in ["torch", "diffusers", "transformers", "xformers", "PIL", "einops",
          "numpy", "cv2", "open_clip", "kornia", "plyfile", "omegaconf",
          "diff_gaussian_rasterization", "simple_knn"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"ok  {m:32s} {v}")
    except Exception as e:
        print(f"!!  {m:32s} {type(e).__name__}: {str(e)[:80]}")
try:
    import torch
    print(f"cuda available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"!!  cuda check: {e}")
PY

echo "== [$(date -Is)] repair finished"
