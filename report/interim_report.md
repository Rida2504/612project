# TextWorld VR — Interim Report

**MSML 612 Class Project — Group 10**
**Date:** 2026-03-15

## 1. Goal

Generate parallax-correct, browser-explorable 3D environments from a text prompt
that a user can walk around in on a Meta Quest 3, using only open-source models
and a single H100 worth of compute.

## 2. Progress to date

### 2.1 Pipeline scaffolding (Stages 1–4)

A four-stage pipeline is end-to-end runnable on the UMD Zaratan cluster:

- **Stage 1 — Text → 360° panorama:** SDXL base 1.0 + `artificialguybr/360Redmond`
  LoRA (weight 0.8) at 2048×1024, fp16, with a horizontal seam-blend post-process.
- **Stage 2 — Panorama → 3D structure (V1):** 8 perspective views reprojected
  from the panorama at 90° FOV.
- **Stage 3 — 3DGS training (V1):** `gsplat` CUDA trainer with frozen
  300 k Gaussians, SH degree 0, no densification.
- **Stage 4 — VR delivery:** WebXR viewer using `mkkellogg/GaussianSplats3D`.
  The earlier Unity export path was dropped after evaluating standalone-Quest
  build complexity vs. browser delivery.

### 2.2 Comparative corpus

A 10-prompt × 3-seed corpus of indoor scenes (coffee shop, library, gallery,
greenhouse, kitchen, etc.) gives 30 scenes per pipeline configuration.

### 2.3 Initial metrics (V1, 300 k Gaussians, SH=0, no densification)

| Metric | Mean ± Std |
|---|---|
| PSNR (in-sample) | 22.0 ± 2.0 dB |
| SSIM | 0.888 ± 0.026 |
| LPIPS | 0.229 ± 0.041 |
| CLIP (text↔panorama) | 0.337 ± 0.030 |

These are baseline numbers; the V2 trainer with adaptive densification and
SH growth 0→3 is implemented and queued to run on the corpus before the
final report.

## 3. Issues identified

### 3.1 Shared-optical-center parallax problem (under investigation)

Initial Quest 3 testing reveals that V1 scenes look correct from the training
origin but degrade at off-origin viewpoints. Diagnosis: all 8 perspective
views in Stage 2 share the same optical center, so 3DGS has no baseline
parallax signal. The trained Gaussians collapse to view-ray-aligned needles
that encode the scene on a 2D shell rather than a 3D volume. Plan: integrate
**LayerPano3D** (SIGGRAPH 2025) to replace Stage 2/3 with depth-layered
3DGS supervised by visibility masks.

### 3.2 Infrastructure issues resolved

- gsplat JIT cache races between 8 parallel cluster workers — fixed via
  per-worker `TORCH_EXTENSIONS_DIR`.
- `lpips` install clobbering CUDA torch — fixed via `pip install --no-deps`.
- Python 3.13 incompatible with torch+cu121 wheels — pinned to 3.12.9.
- BeeOND not triggered by `--tmp=4000G` — needs `#SBATCH --constraint=beeond`.

## 4. Plan for remaining weeks

1. Run V2 trainer (DefaultStrategy densify/prune + SH 0→3) on the full 30-scene
   corpus.
2. Install LayerPano3D in a side-by-side `lp3d` conda env (Python 3.9,
   torch 2.4.0+cu118).
3. Run LP3D pano-depth stage on the 10-scene corpus as the methodology comparison.
4. Build a held-out novel-view evaluation harness (8 cameras on a 0.3 m
   horizontal circle, CLIP scoring per render).
5. WebXR demo site with scene picker and Quest 3 access.
6. Final report with V1/V2 comparison + LayerPano3D integration results.

## 5. Risk register

| Risk | Mitigation |
|---|---|
| FLUX.1-dev (~23 GB) download time on Zaratan | Pre-stage on login node; depth stage alone is enough for methodology comparison |
| LayerPano3D apt-get build deps not available without sudo | Substitute conda-forge packages where possible; fall back to DA-v2 depth stage |
| Quest 3 standalone FPS too low | Cap Gaussians ≤ 1 M, prefer `.ksplat` over `.ply` |
