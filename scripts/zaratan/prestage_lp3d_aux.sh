#!/bin/bash
# prestage_lp3d_aux.sh — download LP3D's auxiliary models alongside FLUX.
#
# Downloads (all via direct curl per P5's lesson):
#   - OneFormer: shi-labs/oneformer_coco_swin_large (panoptic seg, ~1.5 GB)
#   - SAM ViT-H: sam_vit_h_4b8939.pth via facebook (2.4 GB)
#   - LaMa: advimman/lama (200 MB)
#   - LLaVA-1.5-7b: llava-hf/llava-1.5-7b-hf (13 GB)
#   - DA-v2 ViT-L pth: already have it (E11)

set -uo pipefail
LOG="$HOME/scratch/phase4/textworld-vr/logs/prestage_lp3d_aux.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
exec > "$LOG" 2>&1

echo "== [$(date -Is)] prestage_lp3d_aux on $(hostname)"

HF_HOME="$HOME/scratch/phase4/textworld-vr/shared/hf_cache"
HF_HUB_CACHE="$HF_HOME/hub"
TOKEN=$(cat "$HOME/.cache/huggingface/token")

# Direct-URL downloader (same pattern as P5 v3)
grab_hf() {
    local repo="$1" filename="$2" rev="${3:-main}"
    local repo_key="models--${repo//\//--}"
    local snap_dir="$HF_HUB_CACHE/$repo_key/snapshots/$rev"
    local blob_dir="$HF_HUB_CACHE/$repo_key/blobs"
    local target="$snap_dir/$filename"
    mkdir -p "$(dirname "$target")" "$blob_dir"
    if [ -e "$target" ] && [ "$(stat -c%s "$target" 2>/dev/null)" -gt 0 ]; then
        echo "   skip-exists $repo/$filename"; return 0
    fi
    local url="https://huggingface.co/${repo}/resolve/${rev}/${filename}"
    local tmp="$blob_dir/.dl.$$"
    for i in 1 2 3 4 5; do
        if curl -fsSL --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 3600 \
               -H "Authorization: Bearer $TOKEN" -o "$tmp" "$url"; then
            sha=$(sha256sum "$tmp" | awk '{print $1}')
            mv -f "$tmp" "$blob_dir/$sha"
            ln -sf "../../blobs/$sha" "$target"
            sz=$(stat -c%s "$blob_dir/$sha")
            echo "   ok [$(numfmt --to=iec $sz)] $repo/$filename"
            return 0
        fi
        sleep $((i * 3))
    done
    rm -f "$tmp"
    echo "!! $repo/$filename failed"
    return 1
}

# Resolve revisions
get_rev() {
    curl -sL -H "Authorization: Bearer $TOKEN" \
        "https://huggingface.co/api/models/$1/revision/main" \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('sha','main'))"
}

echo "== OneFormer =="
ONEFORMER_REV=$(get_rev "shi-labs/oneformer_coco_swin_large")
for f in config.json pytorch_model.bin preprocessor_config.json; do
    grab_hf "shi-labs/oneformer_coco_swin_large" "$f" "$ONEFORMER_REV" || true
done

echo "== LLaVA-1.5-7b =="
LLAVA_REV=$(get_rev "llava-hf/llava-1.5-7b-hf")
for f in config.json generation_config.json preprocessor_config.json processor_config.json \
         special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json \
         added_tokens.json model.safetensors.index.json \
         model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    grab_hf "llava-hf/llava-1.5-7b-hf" "$f" "$LLAVA_REV" || true
done

echo "== SAM ViT-H =="
mkdir -p "$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D/checkpoints"
SAM="$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D/checkpoints/sam_vit_h_4b8939.pth"
if [ ! -s "$SAM" ]; then
    curl -fsSL --retry 5 --retry-delay 10 \
         -o "$SAM" \
         "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
         && echo "   ok SAM ($(numfmt --to=iec $(stat -c%s "$SAM")))" \
         || echo "   FAIL SAM"
else
    echo "   skip-exists SAM"
fi

echo "== LaMa =="
LAMA="$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D/checkpoints/big-lama.zip"
if [ ! -s "$LAMA" ]; then
    curl -fsSL --retry 5 --retry-delay 10 \
         -o "$LAMA" \
         "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" \
         && echo "   ok LaMa ($(numfmt --to=iec $(stat -c%s "$LAMA")))" \
         || echo "   FAIL LaMa"
else
    echo "   skip-exists LaMa"
fi

echo "== [$(date -Is)] prestage_lp3d_aux finished"
echo "CACHE_SIZE: $(du -sh "$HF_HUB_CACHE" 2>/dev/null | awk '{print $1}')"
echo "CHECKPOINTS_SIZE: $(du -sh $HOME/scratch/phase4/textworld-vr/shared/LayerPano3D/checkpoints 2>/dev/null | awk '{print $1}')"
echo "== DONE =="
