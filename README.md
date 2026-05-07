# TextWorld VR

Text → explorable 3D scene on Meta Quest 3, for MSML 612 Deep Learning
(Spring 2026, Group 10).

## Current status

Revised after ultraplan research (`../ultraplan-textworld-vr-plan.md`):
- **Stage 1 (text → panorama):** SDXL + 360Redmond LoRA + horizontal seam blend.
- **Stage 2 (panorama → 3D structure):** monocular panoramic depth
  (`depth-anything/Depth-Anything-V2-Small-hf` by default) → colored point
  cloud. Replaces the previous parallax-less "8 reprojected views" path.
- **Stage 3 (3DGS training):** `train_gsplat.py` uses `gsplat` CUDA on
  Zaratan A100 when available; a Mac init-only fallback produces a
  Zaratan-ready depth-initialized `.ply`.
- **Stage 4 (VR delivery):** **WebXR in Quest 3 browser** via
  `mkkellogg/GaussianSplats3D`. Unity dropped.

See `../ultraplan-textworld-vr-findings.md` for SOTA research (DreamScene360,
SceneDreamer360, LayerPano3D, CAT3D, VRSplat, etc.) and `../ultraplan-textworld-vr-plan.md`
for the 6-milestone project plan.

## Quickstart

```bash
pip install -r requirements.txt

# End-to-end on a new prompt (new path — depth-based init + gsplat)
python run_pipeline.py "a cozy Japanese coffee shop" --seed 42 --use-depth-init

# Legacy path (random-init v2 trainer, for comparison)
python run_pipeline.py "a cozy Japanese coffee shop" --seed 42 --legacy

# Batch over the 10-scene corpus
python batch_generate.py --prompts-file scenes.txt --seed 42

# Metrics + CSV comparison
python evaluate.py compare --sanity --out outputs/eval.csv
```

## VR viewer

```bash
python -m http.server 8000
# Then on Quest 3 browser: http://<LAN-IP>:8000/viewer/
# (HTTPS required for WebXR in production — see viewer/README.md)
```

## Directory

- `stage1_panorama/` — text → 360° panorama
- `stage2_multiview/`
  - `pano_depth.py` — **new** panorama → depth → point cloud
  - `extract_views.py` — legacy perspective view extraction (still used for
    photometric supervision alongside depth init)
- `stage3_3dgs/`
  - `train_gsplat.py` — **new** gsplat-backed trainer with depth init
  - `train_3dgs_v2.py` — legacy CPU/MPS rasterizer (used as fallback and
    kept for comparison)
- `stage4_vr/` — export helper (Unity-specific; **deprecated**, kept for
  the legacy comparison arm of the report)
- `viewer/` — mkkellogg/GaussianSplats3D WebXR viewer (Quest 3 browser)
- `evaluate.py` — CLIP / PSNR / SSIM / LPIPS / FID + multi-pipeline CSV
- `scenes.txt` — 10 indoor prompts for the comparative study
- `batch_generate.py` — runs Stages 1-3 over a scene list
- `configs/default.yaml` — pipeline configuration

## Hardware

- Zaratan A100/H100 cluster for heavy training (`gsplat`, `pano_depth` with
  real model weights)
- M4 MacBook (this repo's default device = `mps`) for iteration, viewer dev,
  and pipeline scaffolding
- Meta Quest 3 for runtime VR testing via browser WebXR
