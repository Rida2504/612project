# TextWorld VR

Text → explorable 3D scene on Meta Quest 3, for MSML 612 Deep Learning
(Spring 2026, Group 10).

## TL;DR

We diagnosed a fundamental parallax failure in the standard
"panorama → multi-view → 3DGS" pipeline (all reprojected views share an
optical center, so 3DGS learns view-ray-aligned needles instead of a 3D
volume) and fixed it by integrating **LayerPano3D** (SIGGRAPH 2025), which
splits the panorama into semantic depth layers, inpaints occluded regions
with FLUX-Fill, and trains per-layer Gaussians with visibility masks.

**Headline results** on 9 matched scenes:
- **+2.09× novel-view CLIP** (0.153 → 0.303 mean)
- **11.5× reduction in view-to-view drift** (68.59 → 5.96 mean pixel diff)
- ~9.2 M points/scene vs ~1.0 M for the shell pipeline

See [`report/project_report.md`](report/project_report.md) for the full
final report and [`REPORT.md`](REPORT.md) for the pipeline-evaluation report.

## Pipelines in this repo

Three trained pipeline configurations are shipped:

- **V1 — depth-gsplat (simplified):** SDXL panorama → DA-v2 depth → point
  cloud → gsplat with frozen 300 k Gaussians, SH=0, no densification.
  Baseline.
- **V2 — depth-gsplat (full):** same upstream, plus
  `gsplat.DefaultStrategy` adaptive densify/prune and SH growth 0→3.
  In-sample PSNR 33.55 dB, but novel-view CLIP at the noise floor — the
  parallax bug.
- **Layered (LayerPano3D):** SDXL panorama → 360monodepth tangent-face
  depth → OneFormer + KMeans panoptic layering (3–4 layers) → SAM + LaMa +
  FLUX-Fill back-layer inpainting → per-layer 3DGS with visibility masks.
  **The parallax fix.**

Stage 4 (VR delivery) is **WebXR in the Quest 3 browser** via
`mkkellogg/GaussianSplats3D`. The earlier Unity export path is retained in
`stage4_vr/` but deprecated.

## Quickstart

```bash
pip install -r requirements.txt

# End-to-end on a new prompt — depth-gsplat V2 (default trainer)
python run_pipeline.py "a cozy Japanese coffee shop" --seed 42

# Layered pipeline (the parallax fix; requires the lp3d conda env, see §System)
python run_pipeline.py "a cozy Japanese coffee shop" --seed 42 --pipeline layered

# V1 (simplified) for the trainer ablation
python run_pipeline.py "a cozy Japanese coffee shop" --seed 42 --v1

# Batch over the 10-scene corpus
python batch_generate.py --prompts-file scenes.txt --seed 42

# Metrics + CSV comparison (in-sample)
python evaluate.py compare --sanity --out outputs/eval.csv

# Novel-view CLIP (held out — the metric that exposes the parallax bug)
python scripts/zaratan/eval_novel_view.py \
    --ply outputs/splats/<scene>.ply \
    --prompt "<the original prompt>" \
    --n-views 8 --radius 0.3 \
    --out outputs/novel_view/<scene>.json
```

## VR viewer

```bash
python -m http.server 8000
# Then on Quest 3 browser: http://<LAN-IP>:8000/viewer/
# (HTTPS required for WebXR in production — see viewer/README.md)
```

The viewer's scene dropdown is driven by `viewer/scenes.json` and includes
all three pipelines side-by-side for direct comparison.

## Directory

- `stage1_panorama/` — text → 360° panorama (SDXL + 360Redmond LoRA)
- `stage2_multiview/`
  - `pano_depth.py` — panorama → DA-v2 depth → point cloud
  - `extract_views.py` — perspective view extraction (V1/V2 only;
    photometric supervision)
  - `lp3d_layer_gen.py` — adapter from SDXL panorama → LP3D layered data
- `stage3_3dgs/`
  - `train_gsplat.py` — V2 trainer: gsplat CUDA, DefaultStrategy
    densify/prune, SH degree 0→3
  - `train_3dgs_v2.py` — V1 legacy CPU/MPS rasterizer (fallback +
    comparison)
  - `train_layered.py` — LP3D `run_layerpano.py` wrapper with per-layer
    visibility-masked supervision
- `stage4_vr/` — Unity export helper. **Deprecated**, kept for the legacy
  comparison arm of the report.
- `viewer/` — `mkkellogg/GaussianSplats3D` WebXR viewer (Quest 3 browser)
- `evaluate.py` — CLIP / PSNR / SSIM / LPIPS / FID + multi-pipeline CSV
- `scripts/zaratan/`
  - `eval_novel_view.py`, `parallax_proof.py` — held-out novel-view metrics
  - `run_lp3d_batch.sh`, `run_layerpano_batch.sh` — 8-GPU layered dispatch
  - `merge_layered_plys.py` — concat per-layer .plys into a single WebXR-
    loadable file
  - `depth_layer_fallback.py` — depth-quantile fallback when OneFormer
    returns 0 instances
- `scenes.txt` — 10 indoor prompts for the comparative study
- `batch_generate.py` — runs Stages 1–3 over a scene list
- `configs/default.yaml` — pipeline configuration
- `report/` — final report, plots, raw CSVs, demo screenshots
- `deploy/` — optional Nebius VM + container deploy path (not used for the
  class submission)

## System

- **Cluster:** UMD Zaratan, 2 × gpu-a6 nodes, 8 × H100 80 GB total,
  BeeOND shared tmp.
- **Main env:** Python 3.12.9, torch 2.5.1+cu121, gsplat 1.5.3.
- **LP3D env:** conda env `lp3d`, Python 3.9, torch 2.4.0+cu118,
  cudatoolkit 11.8 + cuda-nvcc 11.8 from conda-forge. Required for the
  layered pipeline; built via
  `scripts/zaratan/install_layerpano3d.sh` + `repair_layerpano3d.sh`.
- **Local iteration:** M4 MacBook with `mps` device for the viewer and
  pipeline scaffolding (the layered trainer requires CUDA).
- **VR:** Meta Quest 3 browser, WebXR.

## Findings & plan docs

- [`../ultraplan-textworld-vr-findings.md`](../ultraplan-textworld-vr-findings.md) — SOTA research (DreamScene360, SceneDreamer360, LayerPano3D, CAT3D, VRSplat, etc.)
- [`../ultraplan-textworld-vr-plan.md`](../ultraplan-textworld-vr-plan.md) — 6-milestone project plan
- [`../ultraplan-pano-parallax-plan.md`](../ultraplan-pano-parallax-plan.md) — the 16-step parallax-fix plan executed in this repo
- [`../nightship-pano-parallax-verify-16.md`](../nightship-pano-parallax-verify-16.md) — parallax-proof verification with side-by-side renders
