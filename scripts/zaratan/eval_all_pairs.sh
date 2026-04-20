#!/bin/bash
# eval_all_pairs.sh — run eval_novel_view.py on v2+layered pair for each of the 9 scenes.
set -u

SCRATCH=/home/yog/scratch/phase4/textworld-vr
V2_DIR=$SCRATCH/outputs_v2/splats
V3_DIR=$SCRATCH/outputs_v3/splats
OUT_DIR=$SCRATCH/outputs_v3/eval
mkdir -p $OUT_DIR

declare -A PROMPTS=(
    ["a_sunlit_modernist_kitchen_with_marble_island_and__s42"]="a sunlit modernist kitchen with marble island and floor-to-ceiling windows"
    ["a_grand_hotel_lobby_with_marble_floors,_chandelier_s42"]="a grand hotel lobby with marble floors, chandelier, and ornate furniture"
    ["a_cozy_Japanese_coffee_shop_with_warm_lighting_and_s42"]="a cozy Japanese coffee shop with warm lighting and wooden counter"
    ["a_cozy_Japanese_coffee_shop_with_warm_lighting_and_s43"]="a cozy Japanese coffee shop with warm lighting and wooden counter"
    ["a_cozy_Japanese_coffee_shop_with_warm_lighting_and_s44"]="a cozy Japanese coffee shop with warm lighting and wooden counter"
    ["a_cyberpunk_noodle_bar_with_holographic_menus_and__s43"]="a cyberpunk noodle bar with holographic menus and neon lighting"
    ["a_vast_library_with_floor-to-ceiling_bookshelves_a_s43"]="a vast library with floor-to-ceiling bookshelves and warm reading lamps"
    ["a_minimalist_zen_spa_with_a_hot_spring_pool_and_ri_s44"]="a minimalist zen spa with a hot spring pool and river stones"
    ["a_neon-lit_gaming_room_with_RGB_PC_setups_and_espo_s43"]="a neon-lit gaming room with RGB PC setups and esports gear"
)

source ~/scratch/phase4/textworld-vr/shared/venv/bin/activate
cd $SCRATCH
export PYTHONUNBUFFERED=1

for slug in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$slug]}"
    echo "== $slug"
    echo "   prompt: $prompt"
    v2_ply="$V2_DIR/${slug}_gsplat.ply"
    v3_ply="$V3_DIR/${slug}_layered.ply"
    v2_out="$OUT_DIR/${slug}_v2.json"
    v3_out="$OUT_DIR/${slug}_layered.json"
    if [ -f "$v2_ply" ] && [ ! -f "$v2_out" ]; then
        CUDA_VISIBLE_DEVICES=0 python code/scripts/zaratan/eval_novel_view.py \
            --ply "$v2_ply" --prompt "$prompt" --n-views 8 --radius 0.3 \
            --out "$v2_out" 2>&1 | tail -3
    fi
    if [ -f "$v3_ply" ] && [ ! -f "$v3_out" ]; then
        CUDA_VISIBLE_DEVICES=0 python code/scripts/zaratan/eval_novel_view.py \
            --ply "$v3_ply" --prompt "$prompt" --n-views 8 --radius 0.3 \
            --out "$v3_out" 2>&1 | tail -3
    fi
done
echo "=== ALL DONE ==="
