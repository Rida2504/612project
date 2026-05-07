# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this repo.
This is the **TextWorld VR** project — text → explorable 3D scene on Quest 3,
for MSML 612 Deep Learning (Spring 2026, Group 10).

## Quick orientation

The full project explanation lives in [`README.md`](README.md) and the final
write-up in [`report/project_report.md`](report/project_report.md). Read those
first if you have not already.

Short version: three pipeline configurations are shipped — V1 (depth-gsplat
simplified), V2 (depth-gsplat full), and Layered (LayerPano3D). V2 has the
highest in-sample PSNR but suffers from a shared-optical-center parallax bug
(novel-view CLIP at noise floor). Layered is the parallax fix and the
headline result.

## Repo layout (where to look first)

| You want to… | Start here |
|---|---|
| Understand the pipeline end-to-end | `run_pipeline.py` |
| Modify the panorama stage | `stage1_panorama/` |
| Modify depth or layered stage | `stage2_multiview/pano_depth.py` or `lp3d_layer_gen.py` |
| Modify the gsplat trainer | `stage3_3dgs/train_gsplat.py` (V2) or `train_layered.py` (Layered) |
| Add an evaluation metric | `evaluate.py` for in-sample, `scripts/zaratan/eval_novel_view.py` for held-out |
| Run on the cluster | `scripts/zaratan/` (sbatch files, batch dispatchers, sync daemons) |
| Deploy to cloud | `deploy/nebius/` (laptop-side) and `deploy/server/` (VM-side) |
| Demo in VR | `viewer/index.html` (local) or `www/viewer/index.html` (cluster login server) |
| Read the report | `report/project_report.md` (final), `REPORT.md` (pipeline-eval), `report/interim_report.md` (mid-semester) |

## Two Python environments — do not cross them

- **`main`** (Python 3.12.9, torch 2.5.1+cu121, gsplat 1.5.3): used by every
  stage *except* the layered pipeline. Build with
  `scripts/zaratan/build_venv.sh`.
- **`lp3d`** (Python 3.9, torch 2.4.0+cu118, conda env): required for
  LayerPano3D. Build with `scripts/zaratan/install_layerpano3d.sh` followed by
  `repair_layerpano3d.sh`. The trainer wrapper in `train_layered.py`
  shells out to `lp3d` rather than importing from it.

Mixing imports across these envs will silently break torch CUDA. Use the
`_run_in_env` helper pattern in `deploy/server/main.py` if you need to
orchestrate both from a single process.

## Conventions

- **Gaussian count caps:** V2 trainer caps at 500 k initial; many scenes
  densify to ~1 M. Layered scenes run ~9 M points across 4 layers.
- **Output paths:** scenes land in `outputs/splats/<prompt_slug>_<seed>_<pipeline>.ply`.
  V2 splats use the `_gsplat` suffix (legacy); layered use `_layered`.
- **Cluster scratch:** durable storage at `~/scratch/phase4/textworld-vr/`
  (BeeGFS, both nodes); fast working copies at `/tmp/$USER/textworld-vr/`
  with periodic rsync via `scripts/zaratan/sync_out.sh`.
- **HF cache:** must be on shared scratch (`shared/hf_cache/`) — re-downloading
  ~80 GB of weights per job is not acceptable.

## Things you *will* trip over (canonical fixes)

- **gsplat JIT race** when ≥2 workers start simultaneously sharing
  `$HOME/.cache/torch_extensions/`. Set per-worker `TORCH_EXTENSIONS_DIR`
  *and* `TORCH_CUDA_ARCH_LIST=9.0` (H100) to skip the rebuild on arch list
  changes.
- **`pip install lpips`** clobbers CUDA torch via its dependency closure.
  Use `pip install --no-deps lpips`.
- **`pip install -e <submodule>`** for diff-gaussian-rasterization and
  simple-knn fails because pip's build-isolation env has no torch. Use
  `--no-build-isolation`.
- **Python 3.13** has no torch+cu121 wheel. Stay on 3.12.9.
- **`--tmp=4000G`** does *not* trigger BeeOND on Zaratan. Use
  `#SBATCH --constraint=beeond`.
- **`ssh ... 'nohup foo &'`** dies on pty close. Use
  `tmux new-session -d -s <name> 'foo'`.
- **OneFormer returns 0 instances** on some panoramic scenes. The fallback
  `scripts/zaratan/depth_layer_fallback.py` writes equal-quantile depth
  masks; layered batch should always invoke it as a backstop.
- **Batch tasks are slower than single-GPU runs** (~520 s vs 78 s) because
  `run_pipeline.py` reloads SDXL per task and parallel workers share BeeGFS
  bandwidth. If batch performance matters, keep the pipeline object loaded
  across tasks in `batch_worker.py`.

## Things to *not* edit without thinking

- `stage4_vr/` — Unity export path, **deprecated**. Kept only for the
  legacy comparison arm of the report. Do not add features here; new VR work
  goes in `viewer/`.
- `Dockerfile`, `entrypoint.sh`, `deploy/nebius/on_vm/build_and_run.sh` —
  fallback for the AI Endpoint deploy path. The primary deploy path is the
  dedicated VM (no docker). See `deploy/nebius/README.md`.
- The two `pipeline_v1_clip.csv` / `pipeline_combined_clip.csv` rows for
  `a_cozy_library_s42` were a pre-corpus smoke-test scene with a truncated
  prompt format and a CLIP computation error. Do not include this row in any
  V1 vs V2 aggregate; the published §5.1 numbers exclude it.

## Build history (for reference, not instructions)

This project was built in 15 numbered milestones (E1–E15). Listed here so
that future agents can locate prior work in `git log`.

**Shipped**
- E1–E5: pre-staging, gsplat trainer, rsync workspace, Stage 1 +
  end-to-end smoke test
- E6: batch V1 (33 splats) and V2 (30 splats)
- E7: CLIP scoring via open_clip; V1, V2, and combined CSVs filled
- E8: compare-config harness + metric overview / per-scene plots
- E9–E10: LayerPano3D install (cu118 conda env) + smoke test on compute node
- E11: LP3D pano-depth on 10 scenes — 10/10 succeeded, ~47 s/scene on H100
- E12–E13: WebXR site built and served from cluster login node on :8765
- E14: report with real numbers — PSNR 22.03 → 33.55 dB, LPIPS 0.229 → 0.059
- E15: final rsync + commits, including final LP3D commit

**Pano-parallax follow-up (16-step plan, all 16 complete)**

The shared-optical-center parallax bug surfaced after the V2 batch and was
addressed in the 16-step plan at
`../ultraplan-pano-parallax-plan.md`. All steps shipped:

1. Verify compute budget ✓
2. Install LP3D deps in main cu121 venv ✓
3. Build diff-gaussian-rasterization for cu121 ✓
4. Build simple-knn for cu121 ✓
5. Download FLUX.1-dev + LP3D LoRA ✓
6. SDXL pano → LP3D layer adapter (`stage2_multiview/lp3d_layer_gen.py`) ✓
7. Layered 3DGS trainer wrapper (`stage3_3dgs/train_layered.py`) ✓
8. CLI wiring in `run_pipeline.py` (`--pipeline layered`) ✓
9. Batch 10-scene layered run on 8 H100s ✓
10. Wait-for-completion monitor ✓
11. Held-out novel-view eval harness (`scripts/zaratan/eval_novel_view.py`) ✓
12. Batch eval all 3 pipelines on novel views ✓
13. WebXR update for layered splats ✓
14. Playwright visual regression 3×2 ✓
15. Report update with layered numbers ✓
16. Runtime parallax proof (`scripts/zaratan/parallax_proof.py`) ✓

Verification artefact: `../nightship-pano-parallax-verify-16.md`.
Headline result: V2 novel-view CLIP 0.153 → Layered 0.303 (2.09×); V2
parallax drift 68.59 → Layered 5.96 (11.5× reduction).

## Open items / known gaps

- **`.ksplat` conversion** for the 2 GB layered .plys is not in the build
  pipeline; large .plys can render black in the mkkellogg WebXR viewer.
  Tracked at `deploy/server/README.md` "TODO-ksplat".
- **Metric-scale depth** is non-trivial and out of scope; current scenes use
  relative scale, which is fine for browser exploration but means physical
  VR locomotion ("walk one meter, move one meter in-scene") is not
  calibrated.
- **Cloud deploy hardening** (real queue, JWT auth, HTTPS custom domain) is
  scaffolded in `deploy/server/main.py` but not production-ready. Single
  bearer token + in-memory queue is sufficient for the class submission.
