# TextWorld VR — Text-to-Explorable-3D-World-on-Quest

**MSML 612 Class Project — Final Report**
**Group 10**
**Authors:** Yog Dharaskar, Rida Kutty, Vineet, Keerthi Thota, Sri Akash Kadali
**Contributions:**
- [Name 1]: Stage 1 panorama generation, SDXL/360Redmond LoRA tuning, seam-blend post-process
- [Name 2]: Stage 2 depth + LayerPano3D integration, Zaratan cluster ops, batch dispatcher
- [Name 3]: Stage 3 gsplat trainer (V1/V2), evaluation harness, WebXR viewer + report

**Date:** 2026-04-18

---

## 1. Problem

Generate immersive, *parallax-correct* 3D environments from a text prompt that a
user can explore in a browser-based WebXR viewer on a Meta Quest 3 headset.
Constraint: ~10 weeks, open-source only, no paid APIs, single H100 worth of
compute per scene.

The naive baseline (text → equirectangular panorama → perspective multi-views
→ 3D Gaussian Splatting trained on those views) has a well-known failure mode:
all multi-views share the same optical center, so the trained Gaussians have
no parallax signal. The reconstruction looks flat — or worse, collapses into
view-ray-aligned needles — when viewed from any position other than the
synthetic camera origin. Diagnosing and fixing this failure mode is the
central technical contribution of this report.

## 2. Related Work

| System | Year | Approach | Strengths | Gaps for our constraints |
|--------|------|----------|-----------|--------------------------|
| DreamScene360 | ECCV 2024 | Pano + monocular depth + single-pass 3DGS | Single forward pass, fast | Parallax limited to near-plane |
| LayerPano3D | SIGGRAPH 2025 | Pano + **layered** 3DGS (foreground, mid, background) with FLUX-Fill back-layer inpainting | Strongest perceptual depth to date | cu118/torch 2.4; compiled ceres+pybind11+360monodepth submodules; non-trivial install |
| SceneDreamer360 | 2024 | Pano + NeRF refine | Good 360° coverage | NeRF training slow |
| CAT3D | 2024 | Multi-view diffusion + NeRF | Novel views look real | Requires trained multi-view diffusion model |
| VRSplat | 2024 | Foveated splat rendering | Mobile VR optimized | Needs an input scene |
| **Our pipeline** | 2026 | Pano + **panoramic monocular depth → 3D point cloud → gsplat training**, plus full LayerPano3D layered integration | Diagnoses + fixes shared-optical-center parallax failure; works offline on H100 | Less polished than LayerPano3D layered masks |

## 3. Method

### 3.1 The shared-optical-center parallax bug

Our V1/V2 "depth-gsplat" pipeline generates N perspective views from a single
panorama by reprojecting tangent images. **All N cameras share the same optical
center.** 3DGS training on those views is under-constrained: with no baseline
between cameras, the loss has no gradient signal for *where along each view ray*
a Gaussian should sit. SGD converges to elongated "needle" Gaussians aligned
with the view ray, which perfectly reconstruct the training images
(PSNR → ∞ on training views) but encode the scene on a 2D shell, not a 3D
volume.

From the training origin the shell looks correct. From any off-origin viewpoint
it collapses into smear. This is visible in Quest 3 (the user moves their head
10 cm and the room falls apart) and in any evaluation that uses novel views.
Confusingly, PSNR/SSIM computed on training views stayed high — a textbook
case of SGD memorization, not reconstruction.

### 3.2 Pipeline A (depth-gsplat, ours)

```
prompt  ─▶ SDXL base + 360Redmond LoRA ─▶ 2048×1024 equirect panorama
                                                     │
                                                     ├─▶ perspective view extractor (8 views × 90° FOV)
                                                     │                                    │
                                                     └─▶ Depth Anything V2 (equirect)     │
                                                             │                            │
                                                             └─▶ back-project to 3D  ─────┤
                                                                   (500k points, RGB)     │
                                                                                          ▼
                                                        depth-initialized Gaussians ─▶ gsplat CUDA training
                                                          (Adam + L1 + 0.2·SSIM,
                                                           5k iters, SH degree 0→3,
                                                           adaptive densify/prune)
                                                                                          │
                                                                                          ▼
                                                                                   INRIA .ply (mkkellogg viewer)
```

The depth-initialized Gaussians have plausible 3D positions from pixel 1.
Training refines color/geometry, but parallax is baked in from the monocular
depth prior, not extracted from near-degenerate multi-view geometry.
**Critically, this still produces a "shell" reconstruction** because the
photometric supervision uses only shared-center perspective views. Pipeline A
is therefore best understood as a strong V1/V2 baseline; the actual fix is
Pipeline B.

### 3.3 Pipeline V1 vs V2 — trainer ablation

To quantify the value of adaptive densification + spherical harmonics growth,
we ran two identical pipelines that differ only in the final gsplat training
stage:

| | V1 (simplified) | V2 (full gsplat) |
|---|---|---|
| SH degree | 0 (RGB only) | 0 → 3 (grown every 1000 iter) |
| Densification | disabled | `gsplat.DefaultStrategy` (refine/reset/prune) |
| Max Gaussians | 300 k fixed | up to 500 k (adaptive; many scenes converge to ~1 M) |
| Iters | 5000 | 5000 |

Everything upstream (panorama, depth, perspective-view extraction) is shared,
so PSNR / SSIM / LPIPS differences attribute cleanly to the trainer.

### 3.4 Pipeline B (LayerPano3D, the parallax fix)

Rather than fighting the shared-center limitation, we reformulate the
panorama-to-3D problem as **stacked 2D manifolds** using LayerPano3D
(SIGGRAPH 2025), installed in a side-by-side conda env `lp3d` (py=3.9,
torch 2.4.0+cu118; see Section 8).

1. **Monocular depth** (`gen_panodepth` → 360monodepth + DA-v2 ViT-L) produces
   a tangent-face-aligned disparity map for the equirect panorama.
2. **Panoptic segmentation** (`gen_autolayering` → OneFormer + KMeans over
   depth) carves the scene into 3–4 semantic layers from "front furniture" to
   "back walls + ceiling + floor". When OneFormer returns 0 instances on a
   scene, our `depth_layer_fallback.py` substitutes equal-quantile depth
   masks so the pipeline never stalls.
3. **Back-layer inpainting** (`gen_layerdata` → SAM + LaMa pre-fill +
   FLUX-Fill 30-step inpaint + LLaVA auto-caption + pano LoRA) generates
   plausible content behind occluders so the deeper layers contain real
   structure rather than holes.
4. **Per-layer 3DGS** (`run_layerpano.py`) trains Gaussians per layer with
   visibility masks. Each layer only contributes to renders where its mask
   says it should — the loss has real geometric gradient because different
   views look at different points on different layers.

End-to-end integration in this repo:

| Stage | File | Role |
|---|---|---|
| Adapter | `stage2_multiview/lp3d_layer_gen.py` | SDXL pano → LP3D layered data |
| Trainer | `stage3_3dgs/train_layered.py` | Wrap `run_layerpano.py`, merge layered plys |
| Batch | `scripts/zaratan/run_lp3d_batch.sh` + `run_layerpano_batch.sh` | 8-GPU dispatch |
| Fallback | `scripts/zaratan/depth_layer_fallback.py` | Depth-quantile layers when OneFormer gets 0 instances |
| Eval | `scripts/zaratan/parallax_proof.py` + `eval_novel_view.py` | Novel-view render + CLIP |
| Merge | `scripts/zaratan/merge_layered_plys.py` | Concat 4 layers → 1 .ply for WebXR |

## 4. Evaluation protocol

Corpus: 10 indoor prompts × 3 seeds = 30 scenes per pipeline.

### 4.1 In-sample metrics (`evaluate.py compare`)

- **CLIP score** (open_clip ViT-B/32, laion2b_s34b_b79k): cosine similarity
  between the prompt text embedding and the generated panorama image embedding.
  Range ≈ [0.15, 0.40] for meaningful matches.
- **PSNR / SSIM**: render-vs-GT on the 8 perspective training views.
- **LPIPS (AlexNet)**: learned perceptual distance; lower is better.
- **Gaussian count**: final model size.

### 4.2 Held-out novel-view metrics (`scripts/zaratan/eval_novel_view.py`)

To measure parallax correctness rather than training-view memorization:

- **Novel-view CLIP**: render from 8 cameras on a horizontal circle of
  radius 0.3 m around the scene origin, score each render against the original
  prompt with open_clip ViT-B/32, average.
- **Parallax stability** (`parallax_proof.py`): translate the camera 1 m at
  origin looking at +Z; mean absolute per-pixel difference between the origin
  render and the offset render. Low values = stable 3D reconstruction; high
  values = view-dependent shell that smears under translation.

## 5. Results

### 5.1 Per-pipeline aggregate

30 splats per pipeline (10 prompts × 3 seeds).

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | CLIP (pano) ↑ | Gaussians |
|---|---|---|---|---|---|
| V1 — depth-gsplat (SH=0, no densify) | 22.03 ± 2.00 | 0.888 ± 0.026 | 0.229 ± 0.041 | 0.337 ± 0.030 | 300 000 |
| V2 — depth-gsplat (SH 0→3 + densify) | **33.55 ± 5.25** | **0.976 ± 0.010** | **0.059 ± 0.028** | 0.337 ± 0.030 | 1 065 023 ± 241 939 |
| Δ (V2 − V1) | +11.52 dB | +0.088 | −0.170 | ≈ 0 (same panos) | +3.55× |

Adding spherical-harmonics growth and adaptive densification raises in-sample
PSNR by 11.5 dB and cuts LPIPS by 74% without touching anything upstream.
**However, both V1 and V2 share the parallax bug from §3.1** — high
training-view PSNR is partly memorization. The next subsection shows why this
matters.

![Metrics overview](figures/metrics_overview.png)

See also [figures/per_scene.png](figures/per_scene.png) for per-prompt bars.

### 5.2 Novel-view evaluation — the bug exposed, the fix validated

8 held-out cameras per scene on a 0.3 m horizontal circle around origin. Mean
novel-view CLIP across 9 matched scene pairs:

| Scene | V2 novel-view CLIP | Layered novel-view CLIP | Ratio |
|---|---|---|---|
| cozy_Japanese_coffee_shop_s42 | 0.127 | **0.336** | 2.64× |
| cozy_Japanese_coffee_shop_s43 | 0.123 | **0.350** | 2.83× |
| cozy_Japanese_coffee_shop_s44 | 0.135 | **0.336** | 2.49× |
| cyberpunk_noodle_bar_s43 | 0.192 | **0.299** | 1.56× |
| grand_hotel_lobby_s42 | 0.119 | **0.224** | 1.88× |
| minimalist_zen_spa_s44 | 0.138 | **0.299** | 2.16× |
| neon-lit_gaming_room_s43 | 0.245 | **0.334** | 1.36× |
| sunlit_modernist_kitchen_s42 | 0.117 | **0.280** | 2.39× |
| vast_library_s43 | 0.180 | **0.267** | 1.48× |
| **mean (9 scenes)** | **0.153** | **0.303** | **2.09×** |

V2's novel-view CLIP of 0.153 is at the noise floor (CLIP 0.12–0.19 is
indistinguishable from random for these prompts), confirming that the
high-PSNR training-view fit is shell memorization. The layered pipeline more
than doubles novel-view CLIP, landing in a range comparable to V2's
training-view CLIP — i.e. the layered scene actually *looks like* the prompt
from arbitrary viewpoints.

### 5.3 Parallax stability proof

Sample scene `sunlit_modernist_kitchen_s42`. Camera translated 1 m at origin
looking at +Z; mean absolute per-pixel difference (0–255) between origin
render and offset render:

| Pipeline | Pixel diff | Interpretation |
|---|---|---|
| V2 (shell) | **68.59** | Smear *rotation* — different noise pattern at offset |
| Layered | **5.96** | Real parallax on a stable 3D scene — **11.5× reduction** |

Side-by-side renders are in `report/figures/parallax_proof/`: V2 renders as
unintelligible blobs, while the layered pipeline shows a coherent modernist
kitchen with marble counter, windows, and trees visible through the window in
both views.

### 5.4 Scene density

| Pipeline | Points per scene | .ply size |
|---|---|---|
| V2 (depth-gsplat shell) | ~1.0 M | ~200 MB |
| Layered (LP3D 4-layer) | ~9.2 M | ~2.0 GB |

The 9× density gap reflects layered training producing genuinely volumetric
content rather than a thin shell.

### 5.5 In-sample breakdown (V2)

V2 PSNR ranges from 25.0 dB (art gallery s44 — detailed high-frequency walls)
to 42.7 dB (Japanese coffee shop s42 — smooth low-frequency surfaces). LPIPS
mirrors this: best 0.024, worst 0.128. SSIM is uniformly ≥ 0.95. These numbers
are useful as a trainer-quality signal but should not be read as reconstruction
quality — see §5.2.

## 6. Discussion

- **Training-view PSNR is dangerously misleading for shared-center pipelines.**
  Our V2 had 33.5 dB PSNR but 0.153 novel-view CLIP — essentially noise.
  Reviewers of pano-to-3D work should always demand a held-out novel-view
  metric.
- **Densification matters more than SH alone.** An early simplified V1
  (frozen 300 k Gaussians, RGB-only) was only a minor improvement over a
  random point cloud. Switching to `DefaultStrategy` — which splits/clones
  high-gradient Gaussians and prunes near-transparent ones — is what recovered
  the sharp surfaces that pushed PSNR past 30 dB. SH growth 0→3 adds roughly
  1–2 dB on top.
- **The layered-3DGS fix is structural, not a hyperparameter tweak.** No amount
  of trainer tuning fixes the V2 shell because the supervision itself lacks
  parallax. LayerPano3D fixes it by (a) supervising different layers
  separately and (b) introducing real geometric structure via FLUX-Fill
  inpainted back layers.
- **CLIP on the panorama vs CLIP on novel views are different metrics.** The
  former measures Stage 1 quality only; the latter measures end-to-end scene
  quality. We report both.

## 7. Limitations & Future Work

- **Monocular-depth scale is non-metric;** we use relative scale. Real VR
  locomotion ("stepping into" the scene at the right physical size) would
  benefit from metric scale, achievable by calibrating to a known object size
  or a stereo cue.
- **2 GB layered .ply files stress the mkkellogg WebXR parser.** A
  `.ksplat` post-processing step would materially improve Quest 3 load times;
  it requires a Node toolchain we did not include in this submission.
- **Advanced VR rendering** (foveation, streaming, predictive prefetch) for
  Quest is left to future work.
- **Layered-pipeline novel-view CLIP of 0.30** is good but still well below
  the panorama CLIP of 0.34. Closing this gap would require either better
  layered geometry or a multi-view diffusion refinement pass à la CAT3D.

## 8. System / Reproducibility

| Piece | Detail |
|---|---|
| Cluster | UMD Zaratan — 2 × gpu-a6 nodes, 8 × H100 80 GB, BeeOND shared tmp |
| SLURM | `--nodes=2 --exclusive --gres=gpu:h100:4 --time=23:00:00 --constraint=beeond` |
| Python (main) | 3.12.9 (miniconda) + torch 2.5.1 + cu121 + gsplat 1.5.3 |
| Python (LP3D) | conda env `lp3d`, py 3.9, torch 2.4.0 + cu118, cudatoolkit 11.8 + cuda-nvcc 11.8 from conda-forge |
| Panorama | SDXL base 1.0 + artificialguybr/360Redmond LoRA (weight 0.8), 2048×1024, fp16 |
| Depth | depth-anything/Depth-Anything-V2-Small-hf (ViT-L also cached as .pth) |
| Training (V2) | 5000 iter, up to 500 k Gaussians (many densify to ~1 M), DefaultStrategy densify/prune, SH degree 0→3, per-worker `TORCH_EXTENSIONS_DIR` + `TORCH_CUDA_ARCH_LIST=9.0` |
| Training (Layered) | LP3D `run_layerpano.py` per scene, ~9 M points/scene, FLUX-Fill 30-step inpaint per back layer |
| Total wall time | ~3 h for 30 V2 scenes on 8 H100s; ~14 h for 9 layered scenes |
| Viewer | mkkellogg/gaussian-splats-3d v0.4.7, INRIA-format .ply |

Full build chain: `scripts/zaratan/build_venv.sh` (main venv),
`scripts/zaratan/install_layerpano3d.sh` + `repair_layerpano3d.sh` (LP3D env),
`scripts/zaratan/prestage_models.sh` (HF weights on login),
`sbatch_multinode.sbatch` (job launch),
`scripts/zaratan/run_batch_multigpu.sh` + `batch_worker.py` (8-GPU dispatcher
for V1/V2),
`scripts/zaratan/run_lp3d_batch.sh` + `run_layerpano_batch.sh` (layered
8-GPU dispatcher).

## 9. Deliverables

- **Code:** git tree in this repo — see `git log` for full history.
- **Viewer:** `viewer/index.html` and `www/viewer/index.html` — multi-pipeline
  scene picker with WebXR mode (served on login via
  `python -m http.server 8765 --directory ~/scratch/phase4/textworld-vr/www`).
  Demo screenshots from inside the splat volume (the Quest user's POV):
  - Coffee shop interior ([figures/demo_inside_coffee_shop.png](figures/demo_inside_coffee_shop.png)) — 1.0 M Gaussians (V2)
  - Cyberpunk noodle bar interior ([figures/demo_inside_cyberpunk.png](figures/demo_inside_cyberpunk.png)) — 773 K Gaussians (V2), neon palette
  - Library interior ([figures/demo_inside_library.png](figures/demo_inside_library.png)) — 976 K Gaussians (V2), bookshelves visible
  - Exterior "shell" view for reference: [figures/demo_art_gallery_s43.png](figures/demo_art_gallery_s43.png)
  - Layered comparison renders: [figures/parallax_proof/](figures/parallax_proof/)
- **Report:** this file.
- **Plots:** [figures/metrics_overview.png](figures/metrics_overview.png), [figures/per_scene.png](figures/per_scene.png).
- **Raw CSVs:** [data/pipeline_v1_clip.csv](data/pipeline_v1_clip.csv), [data/pipeline_v2_clip.csv](data/pipeline_v2_clip.csv), [data/pipeline_combined_clip.csv](data/pipeline_combined_clip.csv), [data/novel_view_clip.csv](data/novel_view_clip.csv).
- **Trained splats:** `outputs/splats/*.ply` (V1), `outputs_v2/splats/*.ply`
  (V2), `outputs/splats/*_layered.ply` (Layered) — standard INRIA format,
  also loadable in SuperSplat / gsplat viewers.
- **LP3D pano-depth artefacts:**
  `shared/LayerPano3D/lp3d_smoke_out/layering/{pcd_rgb.ply,depth.npy,rgb.png}`.

---

_Appendix: nightship plan at `nightship-textworld-e2e-plan.md`, ultraplan
findings at `ultraplan-textworld-vr-findings.md`, parallax-proof verification
notes at `nightship-pano-parallax-verify-16.md'
