#!/bin/bash
# prestage_flux.sh — pure-curl download of FLUX.1-dev + LP3D LoRA into HF cache layout.
#
# Rationale: Zaratan login node's Python requests.Session hits SSLEOFError on
# HEAD pre-flights to huggingface.co. Direct curl works (HTTP 200, ~70ms).
# So we curl each file into the correct HF hub cache path, plus symlink dance
# the hub cache expects: hub/models--ORG--REPO/blobs/<sha> + snapshots/<rev>/file.

set -uo pipefail

LOG="$HOME/scratch/phase4/textworld-vr/logs/prestage_flux.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
exec > "$LOG" 2>&1

echo "== [$(date -Is)] prestage_flux v3 (curl) on $(hostname)"

HF_HOME="$HOME/scratch/phase4/textworld-vr/shared/hf_cache"
HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo "")
if [ -z "$TOKEN" ]; then
    echo "ERROR: no HF token at ~/.cache/huggingface/token" >&2
    exit 2
fi

# Grab the revision commit from model_index.json once (we get 200 immediately now)
get_rev() {
    local repo="$1"
    curl -sL --retry 5 --retry-delay 3 \
        -H "Authorization: Bearer $TOKEN" \
        "https://huggingface.co/api/models/${repo}/revision/main" \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('sha', d.get('_id','main')))"
}

# Download one file to the HF-hub-cache layout
# Args: <repo> <filename> <rev>
grab() {
    local repo="$1"
    local filename="$2"
    local rev="$3"
    local repo_key="models--${repo//\//--}"
    local snap_dir="$HF_HUB_CACHE/$repo_key/snapshots/$rev"
    local blob_dir="$HF_HUB_CACHE/$repo_key/blobs"
    local target="$snap_dir/$filename"
    local target_dir="$(dirname "$target")"
    mkdir -p "$target_dir" "$blob_dir"
    if [ -e "$target" ] && [ "$(stat -c%s "$target" 2>/dev/null || stat -f%z "$target")" -gt 0 ]; then
        echo "   skip-exists $filename"
        return 0
    fi
    local url="https://huggingface.co/${repo}/resolve/${rev}/${filename}"
    local tmp_blob="$blob_dir/.download.$$"
    # 10 retries with exponential backoff on SSL/partial
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if curl -fsSL --retry 5 --retry-delay 5 --retry-max-time 900 \
               --connect-timeout 30 --max-time 3600 \
               -H "Authorization: Bearer $TOKEN" \
               -o "$tmp_blob" "$url" 2>&1; then
            # Compute sha, move to blobs, symlink
            local sha=$(sha256sum "$tmp_blob" | awk '{print $1}')
            local blob="$blob_dir/$sha"
            mv -f "$tmp_blob" "$blob"
            ln -sf "../../blobs/$sha" "$target"
            local sz=$(stat -c%s "$blob" 2>/dev/null || stat -f%z "$blob")
            echo "   ok [$(printf '%7s' $(numfmt --to=iec $sz))] $filename"
            return 0
        else
            echo "   [$attempt/10] retry $filename"
            sleep $((attempt * 3))
        fi
    done
    rm -f "$tmp_blob"
    echo "!! exhausted retries: $filename"
    return 1
}

FLUX_REV=$(get_rev "black-forest-labs/FLUX.1-dev")
echo "== FLUX rev: $FLUX_REV"
LP3D_REV=$(get_rev "ysmikey/Layerpano3D-FLUX-Panorama-LoRA")
echo "== LP3D rev: $LP3D_REV"

# FLUX files
FLUX_FILES=(
    "model_index.json"
    "ae.safetensors"
    "tokenizer/merges.txt"
    "tokenizer/special_tokens_map.json"
    "tokenizer/tokenizer_config.json"
    "tokenizer/vocab.json"
    "tokenizer_2/special_tokens_map.json"
    "tokenizer_2/spiece.model"
    "tokenizer_2/tokenizer_config.json"
    "text_encoder/config.json"
    "text_encoder/model.safetensors"
    "text_encoder_2/config.json"
    "text_encoder_2/model-00001-of-00002.safetensors"
    "text_encoder_2/model-00002-of-00002.safetensors"
    "text_encoder_2/model.safetensors.index.json"
    "scheduler/scheduler_config.json"
    "vae/config.json"
    "vae/diffusion_pytorch_model.safetensors"
    "transformer/config.json"
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
    "transformer/diffusion_pytorch_model.safetensors.index.json"
)

echo "== downloading FLUX.1-dev (${#FLUX_FILES[@]} files) =="
ok=0; missing=0
for f in "${FLUX_FILES[@]}"; do
    if grab "black-forest-labs/FLUX.1-dev" "$f" "$FLUX_REV"; then
        ok=$((ok+1))
    else
        missing=$((missing+1))
    fi
done
echo "== FLUX summary: $ok ok, $missing missing =="

# LP3D LoRA — try a few candidate filenames
echo "== downloading LP3D LoRA =="
LORA_FILES=(
    "lp3d_flux.safetensors"
    "lp3d-flux.safetensors"
    "layerpano3d-flux.safetensors"
    "pytorch_lora_weights.safetensors"
    "flux_lora.safetensors"
)
lora_ok=0
for f in "${LORA_FILES[@]}"; do
    if grab "ysmikey/Layerpano3D-FLUX-Panorama-LoRA" "$f" "$LP3D_REV" 2>/dev/null; then
        lora_ok=$((lora_ok+1))
    fi
done
echo "== LP3D LoRA: $lora_ok / ${#LORA_FILES[@]} filenames matched"

# Verify via huggingface_hub snapshot lookup (should no longer need to download)
source "$HOME/scratch/phase4/textworld-vr/shared/venv/bin/activate"
python - <<PY
import os
os.environ["HF_HOME"] = "$HF_HOME"
os.environ["HF_HUB_CACHE"] = "$HF_HUB_CACHE"
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import try_to_load_from_cache
n_ok = 0
for f in """${FLUX_FILES[@]}""".split():
    p = try_to_load_from_cache("black-forest-labs/FLUX.1-dev", f)
    if p: n_ok += 1
print(f"cache-lookup ok: {n_ok}/{len(${#FLUX_FILES[@]})}")
PY

echo "== [$(date -Is)] prestage_flux v3 finished"
echo "CACHE_SIZE: $(du -sh "$HF_HUB_CACHE" 2>/dev/null | awk '{print $1}')"
echo "== DONE =="
