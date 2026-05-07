#!/bin/bash
# run_lp3d_panodepth.sh — run LayerPano3D's gen_panodepth.py over our 10-scene corpus.
#
# Produces an LP3D-style layered point cloud (360monodepth tangent-face depth, not
# our per-equirect DA-v2) for each scene. This is the depth/point-cloud stage of
# LayerPano3D; full scene generation additionally requires FLUX weights (~23 GB)
# and layered inpainting, which are deferred.
#
# Inputs: panoramas from ~/scratch/phase4/textworld-vr/outputs_v2/panoramas/*_s42.png
# Outputs: ~/scratch/phase4/textworld-vr/outputs/lp3d_pcd/<prompt-slug>_s42/layering/pcd_rgb.ply
#
# Must run on compute node (H100).

set -uo pipefail

LP3D="$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D"
PANO_DIR="$HOME/scratch/phase4/textworld-vr/outputs_v2/panoramas"
OUT_ROOT="$HOME/scratch/phase4/textworld-vr/outputs/lp3d_pcd"
LOG="$HOME/scratch/phase4/textworld-vr/logs/lp3d_panodepth_batch.log"

mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"
exec > "$LOG" 2>&1

echo "== [$(date -Is)] lp3d panodepth batch starting on $(hostname)"

source "$HOME/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate lp3d

# Runtime env: match smoke that succeeded
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libglog.so"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$LP3D"

n_ok=0
n_fail=0
n_skip=0
for pano in "$PANO_DIR"/*_s42.png; do
    [ -f "$pano" ] || continue
    base="$(basename "$pano" .png)"
    out_dir="$OUT_ROOT/$base"
    pcd="$out_dir/layering/pcd_rgb.ply"
    if [ -f "$pcd" ]; then
        echo "skip-exists: $base"
        n_skip=$((n_skip+1))
        continue
    fi
    echo "== [$(date -Is)] panodepth: $base"
    mkdir -p "$out_dir"
    if python gen_panodepth.py --input_path "$pano" --save_dir "$out_dir" 2>&1 | tail -20; then
        if [ -f "$pcd" ]; then
            echo "ok: $base ($(stat -c%s "$pcd" 2>/dev/null || stat -f%z "$pcd") bytes)"
            n_ok=$((n_ok+1))
        else
            echo "FAIL: $base — script exited 0 but no pcd_rgb.ply"
            n_fail=$((n_fail+1))
        fi
    else
        echo "FAIL: $base"
        n_fail=$((n_fail+1))
    fi
done

echo
echo "== [$(date -Is)] batch complete: ok=$n_ok fail=$n_fail skip=$n_skip"
echo "== outputs: $(find "$OUT_ROOT" -name 'pcd_rgb.ply' | wc -l) pcd_rgb.ply files"
echo "== DONE =="
