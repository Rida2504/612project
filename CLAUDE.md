## Nightship Status
Tag: textworld-e2e | Step: E6b/E9 (parallel) | Phase: batch+install
Working dir: /Users/yog/PRG/College/612/Project/textworld-vr
Plan: /Users/yog/PRG/College/612/Project/nightship-textworld-e2e-plan.md
Cluster: job 19035389 running on gpu-a6-[8-9] (~20h remaining)

## In-flight work
- Batch v2 running on all 8 H100s (per-worker isolated TORCH_EXTENSIONS_DIR)
  17/30 splats on disk (23 completed tasks counting skip-exists + ok). ETA ~25 min more.
- LayerPano3D install in tmux lp3dinstall on login node (conda env lp3d, cu118 torch)
  Currently at step "installing torch 2.4.0 + cu118". ETA ~10-20 min more.

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
