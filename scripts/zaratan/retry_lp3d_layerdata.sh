#!/bin/bash
# retry_lp3d_layerdata.sh — re-run gen_layerdata ONE scene at a time on a given GPU.
# Used to recover from OOM kills caused by 4 parallel FLUX-Fill processes on one node.
#
# Usage: ./retry_lp3d_layerdata.sh <scene_list.txt> [<gpu_idx>]
# Each scene MUST already have layering/layer{0,1,2}/ populated.
set -u

LIST="${1:?need scene list file}"
GPU_IDX="${2:-0}"
LP3D_ROOT="${LP3D_ROOT:-/home/yog/scratch/phase4/textworld-vr/shared/LayerPano3D}"
LOG_ROOT="${LOG_ROOT:-/home/yog/scratch/phase4/textworld-vr/logs/lp3d_retry}"
mkdir -p "$LOG_ROOT"

source "$HOME/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate lp3d
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libglog.so"
export HF_HOME=/home/yog/scratch/phase4/textworld-vr/shared/hf_cache
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

cd "$LP3D_ROOT"

while IFS= read -r scene_dir; do
    [ -z "$scene_dir" ] && continue
    name=$(basename "$scene_dir")
    log="$LOG_ROOT/$name.log"
    echo "[retry gpu=$GPU_IDX] START $name" | tee -a "$log"
    if [ ! -f "$scene_dir/layering/layer0/layer0_mask.png" ]; then
        echo "[retry gpu=$GPU_IDX] NO layer0_mask.png — running depth-quantile fallback" | tee -a "$log"
        python "/home/yog/scratch/phase4/textworld-vr/scripts/zaratan/depth_layer_fallback.py" \
            --layering-dir "$scene_dir/layering" --n-layers 3 >> "$log" 2>&1
    fi
    # Skip gen_layerdata if every layer already has an inpaint.png (already succeeded previously).
    layers_done=1
    for i in 0 1 2; do
        [ -f "$scene_dir/layering/layer$i/layer${i}_inpaint.png" ] || layers_done=0
    done
    if [ "$layers_done" -eq 1 ]; then
        echo "[retry gpu=$GPU_IDX] layerdata already present — skipping" | tee -a "$log"
    else
        CUDA_VISIBLE_DEVICES="$GPU_IDX" python gen_layerdata.py \
            --base_dir "$scene_dir/layering" >> "$log" 2>&1 \
            && echo "[retry gpu=$GPU_IDX] layerdata OK $name" | tee -a "$log" \
            || { echo "[retry gpu=$GPU_IDX] FAIL layerdata $name" | tee -a "$log"; continue; }
    fi
    # Follow with gen_traindata (Infusion) on the same GPU, still sequential.
    CUDA_VISIBLE_DEVICES="$GPU_IDX" python gen_traindata.py \
        --layerpano_dir "$scene_dir/layering" \
        --save_dir "$scene_dir/layering" \
        --root "$scene_dir/layering" >> "$log" 2>&1 \
        && echo "[retry gpu=$GPU_IDX] traindata OK $name" | tee -a "$log" \
        || { echo "[retry gpu=$GPU_IDX] FAIL traindata $name" | tee -a "$log"; continue; }
    echo "[retry gpu=$GPU_IDX] DONE $name" | tee -a "$log"
done < "$LIST"
echo "[retry gpu=$GPU_IDX] ALL DONE $(date -Is)"
