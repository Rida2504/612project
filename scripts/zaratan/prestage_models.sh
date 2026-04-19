#!/bin/bash
# prestage_models.sh — download HuggingFace models on the LOGIN NODE.
#
# Per Zaratan policy (https://hpcc.umd.edu/kb/filexfer/), compute nodes
# cannot access the public IPv4 internet. IPv6 works in practice but is
# flaky. The proper pattern is: download here, then read from the
# shared BeeGFS cache on compute nodes.
#
# Run this ON THE LOGIN NODE before submitting jobs:
#   ssh ZaratanLogin 'bash ~/scratch/phase4/textworld-vr/code/scripts/zaratan/prestage_models.sh'

set -euo pipefail

SCRATCH_ROOT="$HOME/scratch/phase4/textworld-vr"
HF_CACHE="$SCRATCH_ROOT/shared/hf_cache"
VENV="$SCRATCH_ROOT/shared/venv"

mkdir -p "$HF_CACHE/hub" "$HF_CACHE/transformers"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export TRANSFORMERS_CACHE="$HF_CACHE/transformers"

echo "== pre-staging models to $HF_CACHE"
echo "== host: $(hostname)"

# Need the venv's python so huggingface_hub/transformers imports resolve
if [ ! -d "$VENV" ]; then
    echo "ERROR: venv not built yet. Run scripts/zaratan/build_venv.sh first (on a compute node for CUDA wheels)."
    exit 1
fi
source "$VENV/bin/activate"

MODELS=(
    # Panorama generation
    "stabilityai/stable-diffusion-xl-base-1.0"
    # Panorama LoRA
    "artificialguybr/360Redmond"
    # Depth Anything v2 for panoramic depth (Stage 2)
    "depth-anything/Depth-Anything-V2-Small-hf"
    "depth-anything/Depth-Anything-V2-Base-hf"
    # CLIP for evaluation
    "openai/clip-vit-base-patch32"
)

python - <<PY
import os, sys
from huggingface_hub import snapshot_download

models = [m.strip() for m in """$(printf '%s\n' "${MODELS[@]}")""".strip().splitlines()]
for m in models:
    print(f"== fetching {m}")
    try:
        p = snapshot_download(
            repo_id=m,
            cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            resume_download=True,
        )
        print(f"   -> {p}")
    except Exception as e:
        print(f"   !! failed: {e}", file=sys.stderr)
        # Keep going so one failing model doesn't block others
PY

echo
echo "== cache size =="
du -sh "$HF_CACHE" 2>/dev/null
echo
echo "== contents =="
ls "$HF_CACHE/hub/" 2>/dev/null | head -20
