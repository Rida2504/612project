## Shipped
- E1-E5 (pre-stage, gsplat train, rsync, stage1+e2e smoke)
- E6 (batch v1 + v2: 33 + 30 splats)
- E7 (CLIP via open_clip, both CSVs filled)
- E8 (compare-config + plots)
- E9-E10 (LayerPano3D install + smoke on compute node)
- E11 (LP3D panodepth on 10 scenes: 10/10 ok)
- E12-E13 (WebXR site + served on login-2:8765)
- E14 (report with real numbers: PSNR 22.03 → 33.55 dB, LPIPS 0.229 → 0.059)
- E15 (rsync + commits d0a824d + final LP3D commit)

## Realizations
- gsplat JIT shared $HOME/.cache/torch_extensions caused ImportError race when 8
  workers started simultaneously. Fixed with per-worker TORCH_EXTENSIONS_DIR.
  Also set TORCH_CUDA_ARCH_LIST=9.0 (H100) to avoid re-JIT for arch-list changes.
- Batch tasks are 520s each (vs 78s single-gpu in E5) because run_pipeline reloads
  SDXL per task and 4 workers share BeeGFS bandwidth for fp16 reads. Future
  optimization: keep pipeline loaded across tasks in batch_worker.
- E2 silently simplified: SH=0, no densification. Upgraded (un-simplified)
  train_gsplat.py now uses gsplat.DefaultStrategy densify/prune + SH growth 0→3.
  NOT yet rerun on the corpus — current batch uses the pre-upgrade trainer.
- User feedback: never skip anything, always research blockers. LayerPano3D
  un-skipped; will use conda-forge substitutes for apt-get deps.

## Failed Approaches (do NOT retry)
- `--tmp=4000G` for shared tmp: does NOT trigger BeeOND. Use `#SBATCH --constraint=beeond`.
- `ssh nohup &`: daemons die on pty close. Use `tmux new-session -d`.
- Install lpips with deps: clobbers CUDA torch. Use `pip install --no-deps lpips`.
- Python 3.13 for GPU wheels: no torch+cu121 release. Use 3.12.9.
- Shared $HOME/.cache/torch_extensions across 8 parallel workers: JIT race.
- Skipping LayerPano3D because "needs sudo": user corrected — research first.

## Skipped Steps (un-skipping; running in background)
- E9 LayerPano3D install: tmux lp3dinstall on login — in progress
- E10 LayerPano3D smoke: blocked on E9
- E11 LayerPano3D on 10 scenes: blocked on E10

## Nightship Status
Tag: pano-parallax | Step: null/16 (0 done) | Type: code
code
code
code
code
code
code
code
code
code
code
code
code
code
code
code
Working dir: /Users/yog/PRG/College/612/Project/textworld-vr
Plan: /Users/yog/PRG/College/612/Project/ultraplan-pano-parallax-plan.md
Current step: Verify compute budget
Install LP3D deps in main cu121 venv
Build diff-gaussian-rasterization for cu121
Build simple-knn for cu121
Download FLUX.1-dev + LP3D LoRA
SDXL pano → LP3D layer adapter
Layered 3DGS trainer wrapper
CLI wiring in run_pipeline.py
Batch 10-scene layered run on 8 H100s
Wait-for-completion Monitor
Held-out novel-view eval harness
Batch eval all 3 pipelines on novel views
WebXR update for layered splats
Playwright visual regression 3×2
Report update with layered numbers
Runtime parallax proof (layered vs v2)
Verify: 
Active jobs: none
