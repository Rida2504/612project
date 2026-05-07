#!/bin/bash
# install_lp3d_in_main_venv.sh — install LP3D runtime deps into our cu121 main venv.
#
# Runs on login node (no GPU needed for pip). Skips CUDA extensions (diff-gaussian-
# rasterization, simple-knn) — those are handled by P3/P4 on the compute node.

set -uo pipefail

LOG="$HOME/scratch/phase4/textworld-vr/logs/p2_install_lp3d_deps.log"
mkdir -p "$(dirname "$LOG")"
exec > "$LOG" 2>&1

echo "== [$(date -Is)] P2 starting on $(hostname)"
source "$HOME/scratch/phase4/textworld-vr/shared/venv/bin/activate"
python --version
pip --version

# Core deps that LP3D uses at RUNTIME (inference/layering/scene generation).
# Keep explicit versions where we know them from LP3D's requirements.txt, let
# the rest float with our cu121 torch 2.5.1.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# First pass — PyPI packages only. Install each individually with --no-deps
# so one failure doesn't stop others. Already-installed ones no-op.
PKGS=(
    "segmentation-models-pytorch"
    "basicsr==1.4.2"
    "realesrgan==0.3.0"
    "gfpgan==1.3.8"
    "kornia"
    "einops"
    "omegaconf"
    "hydra-core"
    "pyyaml"
    "tqdm"
    "loguru"
    "imageio"
    "imageio-ffmpeg"
    "opencv-python"
    "scikit-image"
    "plyfile"
    "sentencepiece"
    "safetensors"
    "peft"
    "timm"
    "open_clip_torch"
)
for p in "${PKGS[@]}"; do
    echo "== pip install $p"
    pip install --no-cache-dir --timeout 45 "$p" --upgrade-strategy only-if-needed \
        2>&1 | tail -3 || echo "!! failed: $p"
done

# LightGlue is only on GitHub (cvg/LightGlue)
echo "== pip install lightglue from github"
pip install --no-cache-dir --timeout 60 \
    "git+https://github.com/cvg/LightGlue.git" 2>&1 | tail -5 || echo "!! lightglue failed"

# LP3D imports these — verify
python - <<'PY'
import importlib, sys
mods = ["segmentation_models_pytorch", "lightglue", "basicsr", "realesrgan",
        "gfpgan", "kornia", "einops", "omegaconf", "hydra", "loguru",
        "imageio", "cv2", "skimage", "plyfile", "sentencepiece",
        "safetensors", "peft", "timm", "open_clip"]
ok = 0; fail = []
for m in mods:
    try:
        importlib.import_module(m); ok += 1
    except Exception as e:
        fail.append(f"{m}: {type(e).__name__}: {str(e)[:80]}")
print(f"imports ok: {ok}/{len(mods)}")
for f in fail: print("  FAIL", f)
sys.exit(0 if not fail else 1)
PY

echo "== [$(date -Is)] P2 finished"
echo "== DONE =="
