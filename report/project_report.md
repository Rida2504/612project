# TextWorld VR — Text-to-Explorable-3D-World-on-Quest

**MSML 612 Class Project — Final Report**
**Author:** Yog
**Date:** 2026-04-18

---

## 1. Problem

Generate immersive, *parallax-correct* 3D environments from a text prompt that a
user can explore in a browser-based WebXR viewer on a Meta Quest headset.
Constraint: a single developer, ~10 weeks, open-source only, no paid APIs.

The naive baseline (text → equirectangular panorama → perspective multi-views
→ 3D Gaussian Splatting trained on those views) has a well-known failure mode:
all multi-views share the same optical center, so the trained Gaussians have
no parallax. The reconstruction looks flat when viewed from any position
other than the synthetic camera origin.

## 2. Related Work

| System | Year | Approach | Strengths | Gaps for our constraints |
|--------|------|----------|-----------|--------------------------|
| DreamScene360 | ECCV 2024 | Pano + monocular depth + single-pass 3DGS | Single forward pass, fast | Parallax limited to near-plane |
| LayerPano3D | SIGGRAPH 2025 | Pano + **layered** 3DGS (foreground, mid, background) | Strongest perceptual depth to date | cu118/torch 2.4; compiled ceres+pybind11+360monodepth submodules; non-trivial install |
| SceneDreamer360 | 2024 | Pano + NeRF refine | Good 360° coverage | NeRF training slow |
| CAT3D | 2024 | Multi-view diffusion + NeRF | Novel views look real | Requires trained multi-view diffusion model |
| VRSplat | 2024 | Foveated splat rendering | Mobile VR optimized | Needs an input scene |
| Our pipeline | 2026 | Pano + **panoramic monocular depth → 3D point cloud → gsplat training** | No parallax loss from shared-center views; works offline on H100 | Less polished than LayerPano3D layered masks |

## 3. Method

### 3.1 Pipeline A (primary, ours)

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

Key novelty: the depth-initialized Gaussians already have plausible 3D positions
from pixel 1. Training refines color/geometry but parallax is baked in from the
monocular depth prior, not extracted from near-degenerate multi-view geometry.

### 3.2 Pipeline B (baseline, LayerPano3D SIGGRAPH 2025)

Layered-panorama approach. See Section 5 for install details on Zaratan.

## 4. Evaluation protocol

Corpus: 10 indoor prompts × 3 seeds = 30 scenes per pipeline.

Automated metrics (`evaluate.py compare`):
- **CLIP score** (openai/clip-vit-base-patch32): text-panorama alignment.
- **PSNR / SSIM**: render-vs-GT on the 8 perspective training views (in-sample).
- **LPIPS (AlexNet)**: learned perceptual distance.
- **Gaussian count**: final model size.

We also record training time (wall, s) and final loss.

## 5. Results

### 5.1 Per-pipeline aggregate

_Auto-filled from `outputs/eval/pipeline_a.csv` and `pipeline_b.csv` — see `metrics_overview.png`._

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Gaussians |
|---|---|---|---|---|---|
| depth-gsplat (ours) | TBD | TBD | TBD | TBD | TBD |
| LayerPano3D | TBD | TBD | TBD | TBD | TBD |

### 5.2 Per-scene breakdown

_See `per_scene.png` for bars across all 10 prompts._

## 6. Discussion

Expected findings (will validate once CSVs are generated):
- Depth-init gsplat produces visible parallax that the legacy multi-view v2
  trainer cannot — a qualitative improvement not visible in in-sample PSNR.
- PSNR on in-sample views favors methods with more Gaussians (tautological);
  the fairer measure is LPIPS and a held-out view set.
- LayerPano3D's layered representation should produce the cleanest occlusion
  parallax at the cost of install complexity. If Pipeline B succeeded, the
  comparison is fair because both pipelines are run on the same corpus with
  the same SDXL panorama source.

## 7. Limitations & Future Work

- Our depth path uses DA-v2 applied per-tangent-face of the panorama; the
  LayerPano3D paper's `360monodepth` alignment gives better seam handling.
- Monocular-depth scale is non-metric; we use relative scale. Real VR
  locomotion ("stepping into" the scene) would benefit from metric scale —
  achievable by later calibrating to a known object size or stereo cue.
- Evaluation is in-sample. A held-out novel-view evaluation (render from a
  random camera pose, CLIP with the original prompt) would tell us more about
  generalization.
- The WebXR viewer is a drop-in using mkkellogg/gaussian-splats-3d; advanced
  foveation / streaming for Quest is left to future work.

## 8. System / Reproducibility

| Piece | Detail |
|---|---|
| Cluster | UMD Zaratan — 2 × gpu-a6 nodes, 8 × H100 80GB, BeeOND shared tmp |
| SLURM | `--nodes=2 --exclusive --gres=gpu:h100:4 --time=23:00:00 --constraint=beeond` |
| Python | 3.12.9 (miniconda) + torch 2.5.1 + cu121 + gsplat 1.5.3 |
| Panorama | SDXL base 1.0 + artificialguybr/360Redmond LoRA (weight 0.8) |
| Depth | depth-anything/Depth-Anything-V2-Small-hf (plus V2-Base cached) |
| Training | 5000 iter, 500k max Gaussians, DefaultStrategy densify/prune, SH degree 0→3 |
| Total wall time | ~40 min for 30 scenes on 8 H100s |

Full build instructions: `scripts/zaratan/build_venv.sh` (venv),
`scripts/zaratan/prestage_models.sh` (HF weights, run on login), and
`sbatch_multinode.sbatch` (job launch).

## 9. Deliverables

- **Code:** [github-link-here] — commit hash and full tree
- **Viewer:** `www/viewer/index.html` — 10-scene picker with WebXR mode
- **Report:** this file
- **Plots:** `outputs/eval/metrics_overview.png`, `outputs/eval/per_scene.png`
- **Raw CSVs:** `outputs/eval/pipeline_a.csv` (plus `pipeline_b.csv` if B completed)
- **Trained splats:** `outputs/splats/*.ply` (standard INRIA format; also loadable in SuperSplat)

---

_Appendix: findings file at `ultraplan-textworld-vr-findings.md`, nightship plan at `nightship-textworld-e2e-plan.md`._
