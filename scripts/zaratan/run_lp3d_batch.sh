#!/bin/bash
# run_lp3d_batch.sh — parallelize LP3D pano→layers over the allocated GPUs.
#
# Partitions the pano list into N_GPU chunks (round-robin) and runs each chunk
# on its assigned GPU (serial within a chunk, parallel across chunks).
# Per scene: gen_panodepth → gen_autolayering → gen_layerdata.
#
# Usage:  ./run_lp3d_batch.sh <pano_list.txt> [<gpu_count>]
set -u

LIST="${1:?need pano list file}"
N_GPU="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
N_GPU="${N_GPU:-1}"

LP3D_ROOT="${LP3D_ROOT:-/home/yog/scratch/phase4/textworld-vr/shared/LayerPano3D}"
BEEOND_ROOT="${BEEOND_ROOT:-/scratch/local/19035389}"
LOG_ROOT="${LOG_ROOT:-/home/yog/scratch/phase4/textworld-vr/logs/lp3d_batch}"
mkdir -p "$LOG_ROOT" "$BEEOND_ROOT/scenes"

source "$HOME/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate lp3d
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libglog.so"
export HF_HOME=/home/yog/scratch/phase4/textworld-vr/shared/hf_cache
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

process_scene() {
    local pano="$1"
    local gpu="$2"
    local scene_name=$(basename "$pano" .png)
    local scene_dir="$BEEOND_ROOT/scenes/$scene_name"
    local log="$LOG_ROOT/$scene_name.log"
    mkdir -p "$scene_dir"
    cp -f "$pano" "$scene_dir/rgb.png"
    echo "[gpu=$gpu] START $scene_name" | tee -a "$log"
    # LP3D scripts load checkpoints via RELATIVE paths, so cwd MUST be LP3D_ROOT.
    cd "$LP3D_ROOT" || return 99
    CUDA_VISIBLE_DEVICES="$gpu" python gen_panodepth.py \
        --input_path "$scene_dir/rgb.png" --save_dir "$scene_dir" >> "$log" 2>&1 \
        && echo "[gpu=$gpu] panodepth OK $scene_name" | tee -a "$log" \
        || { echo "[gpu=$gpu] FAIL panodepth $scene_name" | tee -a "$log"; return 1; }
    CUDA_VISIBLE_DEVICES="$gpu" python gen_autolayering.py \
        --input_dir "$scene_dir" --scene_type indoor >> "$log" 2>&1 \
        && echo "[gpu=$gpu] autolayering OK $scene_name" | tee -a "$log" \
        || { echo "[gpu=$gpu] FAIL autolayering $scene_name" | tee -a "$log"; return 2; }
    # Fallback: if OneFormer detected 0 instances, autolayering skipped writing
    # layer dirs. Build depth-quantile layers so gen_layerdata has something to eat.
    if [ ! -f "$scene_dir/layering/layer0/layer0_mask.png" ]; then
        echo "[gpu=$gpu] fallback depth-quantile layers for $scene_name" | tee -a "$log"
        python "/home/yog/scratch/phase4/textworld-vr/scripts/zaratan/depth_layer_fallback.py" \
            --layering-dir "$scene_dir/layering" --n-layers 3 >> "$log" 2>&1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python gen_layerdata.py \
        --base_dir "$scene_dir/layering" >> "$log" 2>&1 \
        && echo "[gpu=$gpu] layerdata OK $scene_name" | tee -a "$log" \
        || { echo "[gpu=$gpu] FAIL layerdata $scene_name" | tee -a "$log"; return 3; }
    # gen_traindata: Infusion-based SD inpainting of back layers (for LayerPano trainer)
    CUDA_VISIBLE_DEVICES="$gpu" python gen_traindata.py \
        --layerpano_dir "$scene_dir/layering" \
        --save_dir "$scene_dir/layering" \
        --root "$scene_dir/layering" >> "$log" 2>&1 \
        && echo "[gpu=$gpu] traindata OK $scene_name" | tee -a "$log" \
        || { echo "[gpu=$gpu] FAIL traindata $scene_name" | tee -a "$log"; return 4; }
    echo "[gpu=$gpu] DONE $scene_name" | tee -a "$log"
}

export -f process_scene
export LP3D_ROOT BEEOND_ROOT LOG_ROOT

# Assign scenes round-robin across GPUs.
mapfile -t PANOS < "$LIST"
echo "[batch] scenes=${#PANOS[@]} gpus=$N_GPU"

for gpu in $(seq 0 $((N_GPU-1))); do
    (
        for i in "${!PANOS[@]}"; do
            if (( i % N_GPU == gpu )); then
                process_scene "${PANOS[$i]}" "$gpu"
            fi
        done
    ) &
done
wait
echo "[batch] ALL DONE  $(date -Is)"
