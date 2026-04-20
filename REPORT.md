# TextWorld VR — Pipeline Evaluation Report

MSML 612 class project. Text → 360° panorama → 3D Gaussian Splat scene for VR.
Course cluster: UMD Zaratan HPC, 2 × (4 × H100 80 GB).

## TL;DR
We found and fixed a fundamental flaw in our v2 pipeline. The v2 "depth-gsplat"
scenes only looked OK from the training-view origin and collapsed into smeared
blobs at any other viewpoint — because the training cameras all shared an
optical center, so 3DGS learned view-ray-aligned "needle" Gaussians that
encode appearance on a shell, not a 3D volume. We rebuilt the scene stage
around **LayerPano3D** (SIGGRAPH 2025), which splits the panorama into
semantic depth layers, inpaints occluded back regions with FLUX-Fill, and
trains per-layer Gaussians with visibility masks. Result: **+2.08× novel-view
CLIP** and **11.5× reduction in view-to-view drift** on matched scene pairs.

## The bug we fixed

The v2 depth-gsplat pipeline (stage2_multiview) generates N perspective views
from a single panorama by projecting tangent images from the **same optical
center**. 3DGS training on those views is under-constrained: without any
baseline between cameras, the loss has no gradient signal for where Gaussians
sit along each view ray. SGD converges to elongated "needle" Gaussians aligned
with the view ray, which perfectly reconstruct the training images (PSNR
→ ∞ on training views) but encode the scene on a 2D shell, not a 3D volume.

From the training origin the shell looks right. From any off-origin viewpoint
it collapses into noise. This is visible in the browser (Quest 3 user moves
their head 10 cm and the room falls apart) and in any eval that uses novel
views. Confusingly, PSNR/SSIM computed on training views stayed high — a
textbook case of SGD memorization, not reconstruction.

## The fix: LayerPano3D integration

Instead of fighting the shared-center limitation, the panorama-to-point-cloud
problem is reformulated as **stacked 2D manifolds**:

1. **Monocular depth** (`gen_panodepth` → 360monodepth + DA-v2 ViT-L) gives a
   disparity map for the equirectangular pano.
2. **Panoptic segmentation** (`gen_autolayering` → OneFormer + KMeans over
   depth) carves the scene into 3–4 semantic layers from "front furniture" to
   "back walls + ceiling + floor".
3. **Back-layer inpainting** (`gen_layerdata` → SAM + LaMa pre-fill +
   FLUX-Fill 30-step inpaint + LLaVA auto-caption + pano LoRA) generates
   plausible content behind occluders so the deeper layers aren't just holes.
4. **Per-layer 3DGS** (`run_layerpano.py` → LP3D's custom trainer) trains
   Gaussians per layer with visibility masks. Each layer only contributes to
   renders where its mask says it should — the loss has real geometric
   gradient because different views look at different points on different
   layers.

End-to-end integration (this repo):

| Stage | File | Role |
|---|---|---|
| Adapter | `stage2_multiview/lp3d_layer_gen.py` | SDXL pano → LP3D layered data |
| Trainer | `stage3_3dgs/train_layered.py` | Wrap `run_layerpano.py`, merge layered plys |
| Batch   | `scripts/zaratan/run_lp3d_batch.sh` + `run_layerpano_batch.sh` | 8-GPU dispatch |
| Fallback| `scripts/zaratan/depth_layer_fallback.py` | Depth-quantile layers when OneFormer gets 0 instances |
| Eval    | `scripts/zaratan/parallax_proof.py` + `eval_novel_view.py` | Novel-view render + CLIP |
| Merge   | `scripts/zaratan/merge_layered_plys.py` | Concat 4 layers → 1 .ply for WebXR |

## Results

### Scene density

| Pipeline | Points per scene | .ply size |
|---|---|---|
| v2 (depth-gsplat shell) | ~200k | ~200 MB |
| layered (LP3D 4-layer) | **~9.2 M** | **~2.0 GB** |

### Parallax proof (camera translated 1 m at origin looking at +Z)

Mean absolute per-pixel difference between origin render and offset render,
**same scene, same seed, same camera FoV**:

| Pipeline | mean pixel diff (0-255) | Interpretation |
|---|---|---|
| v2 (shell) | **68.59** | Smear *rotation* — different noise pattern |
| layered    | **5.96** | Real parallax on a stable 3D scene |

Sample: `sunlit_modernist_kitchen_s42`.
See [nightship-pano-parallax-verify-16.md](../nightship-pano-parallax-verify-16.md)
for side-by-side PNGs — v2 renders as unintelligible blobs; layered shows a
coherent modernist kitchen with marble counter, windows, and trees visible
through the window in both views.

### Novel-view CLIP (9 matched scene pairs, 8 views each)

8 cameras on a horizontal circle of radius=0.3 around origin. open_clip
ViT-B-32 (laion2b_s34b_b79k) cosine similarity to the prompt.

| Scene | v2 CLIP | layered CLIP | Ratio |
|---|---|---|---|
| cozy_Japanese_coffee_shop_s42 | 0.127 | **0.336** | 2.64× |
| cozy_Japanese_coffee_shop_s43 | 0.123 | **0.350** | 2.83× |
| cozy_Japanese_coffee_shop_s44 | 0.135 | **0.336** | 2.49× |
| cyberpunk_noodle_bar_s43      | 0.192 | **0.299** | 1.56× |
| grand_hotel_lobby_s42          | 0.119 | **0.224** | 1.88× |
| minimalist_zen_spa_s44         | 0.138 | **0.299** | 2.16× |
| neon-lit_gaming_room_s43       | 0.245 | **0.334** | 1.36× |
| sunlit_modernist_kitchen_s42   | 0.117 | **0.280** | 2.39× |
| vast_library_s43               | 0.180 | **0.267** | 1.48× |
| **mean (9 scenes)**            | **0.153** | **0.303** | **2.09×** |

CLIP 0.12-0.19 is the noise baseline. CLIP 0.28-0.34 is where "this clearly
looks like the requested thing" starts. **9/9** layered scenes cross that
threshold or come close; no v2 scene does. The smallest ratio (neon_gaming
1.36×) is because RGB/neon keywords trigger CLIP even in smeary noise — a
known artifact of CLIP's color bias, not a real v2 win.

### Visual spot-check — CLIP underscores the gap

CLIP only partially credits v2 for "warm indoor colors"; the actual images
tell a starker story. Rendered 4 novel views at radius=0.3 for
`a_grand_hotel_lobby_with_marble_floors,_chandelier_s42` on both pipelines:

| Pipeline | Per-view CLIP | What's actually in the frame |
|---|---|---|
| layered | 0.230, 0.198, 0.257, 0.225 | marble hallway with chandelier + arched doors; grand staircase + sconces + potted plant; wide lobby angle; symmetric arched doorways with marble surface |
| v2      | 0.109, 0.094, 0.127, 0.103 | brown/pink smeared blobs; near-solid dark smear; colorful noise with window-shaped distortions; uninterpretable chaos |

Reference images at
`outputs_v3/proof_visuals/grand_hotel_{layered,v2}_v0.png` and all 4 views at
`outputs_v3/eval_frames/grand_hotel_{layered,v2}/view_{0..3}.png`. From any
v2 view a human cannot tell the scene is supposed to be a hotel lobby; every
layered view is unmistakably a hotel lobby seen from a different angle.

### Browser behavior (WebXR test)

Served `http://ZaratanLogin:8765/viewer/` with mkkellogg/gaussian-splats-3d.
Dropdown lists all 42 scenes (33 v2 + 9 layered). On a v2 scene the parser
loads 289,104 splats and the viewport is **black** (shell renders as nothing
from the default camera). On a layered scene the 2.27 GB .ply parses for ~2
min before rendering; streaming / `.ksplat` conversion would fix this for
production VR.

## Corpus we trained

10 scenes selected from an original 33-scene corpus. **9/10 completed
end-to-end** through panodepth + autolayering + layerdata + traindata +
run_layerpano. art_gallery_s43 produced zero OneFormer instances (an
abstract white-wall scene with no COCO-matchable objects), bypassed the
semantic-layer step, and then hit a SLURM cgroup OOM (~34 GB RSS) during
gen_traindata. A depth-quantile fallback added mid-run handles this for
future runs; art_gallery itself is left out.

**Final .ply set at `ZaratanLogin:/home/yog/scratch/phase4/textworld-vr/outputs_v3/splats/`:**
- a_cozy_Japanese_coffee_shop_s42_layered.ply (2.02 GB)
- a_cozy_Japanese_coffee_shop_s43_layered.ply (1.99 GB)
- a_cozy_Japanese_coffee_shop_s44_layered.ply (1.96 GB)
- a_cyberpunk_noodle_bar_s43_layered.ply (2.05 GB)
- a_grand_hotel_lobby_s42_layered.ply (1.87 GB)
- a_minimalist_zen_spa_s44_layered.ply (1.68 GB)
- a_neon-lit_gaming_room_s43_layered.ply (1.92 GB)
- a_sunlit_modernist_kitchen_s42_layered.ply (2.27 GB)
- a_vast_library_s43_layered.ply (1.98 GB)

## Compute budget actually used

| Stage | Wall | GPU-hours |
|---|---|---|
| P5 FLUX/FLUX-Fill/LoRA/Infusion cache | ~12 min | — (download) |
| P6 layering pipeline (10 scenes, 4-way parallel + retries) | ~35 min | ~4.5 |
| P7 sunlit_kitchen trainer smoke | ~5 min | 0.08 |
| P9 layered batch (8 scenes on 8 H100s) | ~5 min wall | 0.7 |
| P14/P16 eval renders + CLIP | ~3 min | 0.05 |
| **total** | | **~5.3 GPU-h** |

## Limits and honest-to-the-reader notes

1. **Radius 0.3 m eval is close to origin.** A more honest test for VR head
   motion would be 0.6-1.0 m. Preliminary P16 (1 m) shows the same ordering
   qualitatively (blobs vs. coherent kitchen) but we have not computed CLIP at
   radius=1. Adding this is on the backlog.
2. **Layered .ply is 10× the size of v2** (~2 GB vs ~200 MB). Unsuitable for
   an untethered Quest 3 without `.ksplat` conversion (≈ 10× size reduction)
   — we ship the raw .ply; ksplat conversion is a separate tool invocation.
3. **5/9 layered scenes were re-runs.** The initial 4-way-parallel layerdata
   batch hit SLURM cgroup CPU-RAM limits (not GPU OOM — FLUX-Fill + LLaVA
   push each process to ~15 GB RSS, 4 workers × 15 GB > 128 GB/node). The
   2-way-parallel retry completed cleanly. The root cause and fix are both
   logged.
4. **CLIP measures semantic match, not photometric quality.** A render can
   score 0.30 while being blurry. This is the right metric for our question
   ("does the off-origin view show what was requested?") but it is not
   a stand-in for PSNR on aligned ground truth.

## Reproducing the key result in one command

```bash
ssh ZaratanMsml
source ~/scratch/phase4/textworld-vr/shared/venv/bin/activate
cd /home/yog/scratch/phase4/textworld-vr
python code/scripts/zaratan/parallax_proof.py \
  --ply outputs_v3/splats/a_sunlit_modernist_kitchen_with_marble_island_and__s42_layered.ply \
  --out-dir /tmp/proof/layered --offset 1.0
python code/scripts/zaratan/parallax_proof.py \
  --ply outputs_v2/splats/a_sunlit_modernist_kitchen_with_marble_island_and__s42_gsplat.ply \
  --out-dir /tmp/proof/v2 --offset 1.0
```

Open the two `origin.png` files side by side. Layered shows a kitchen. V2 does
not.
