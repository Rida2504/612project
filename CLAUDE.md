## Nightship Status
Tag: textworld-e2e | Step: E1/15 | Phase: foundation
Working dir: /Users/yog/PRG/College/612/Project/textworld-vr
Plan: /Users/yog/PRG/College/612/Project/nightship-textworld-e2e-plan.md
Cluster: job 19035389 running on gpu-a6-[8-9] (~22h remaining)

## Realizations
- BeeOND mounted on both nodes at /scratch/local/19035389 (24 TB shared)
- Python venv at ~/scratch/phase4/textworld-vr/shared/venv (python 3.12.9, torch 2.5.1+cu121, gsplat)
- Pre-staged HF cache currently has: DA-v2 Small (~150 MB), CLIP ViT-B/32. Missing: SDXL base, 360Redmond LoRA, DA-v2 Base.
- sync_out tmux session running on gpu-a6-8 → ~/scratch/outputs/ every 60s
- Compute nodes have NO IPv4 internet; pre-staging on login is the proper pattern
- Gave up on v2 differentiable trainer — gsplat CUDA path is default now; v2 stays as scaffold fallback

## Failed Approaches (do NOT retry)
- `--tmp=4000G` for shared tmp: does NOT trigger BeeOND. Must use `#SBATCH --constraint=beeond`.
- `ssh nohup & disown`: daemons died on gpu-a6-9 pty close. Use `tmux new-session -d` instead.
- Install lpips with deps: clobbers CUDA torch. Must use `pip install --no-deps lpips`.
- Python 3.13 for GPU wheels: no torch+cu121 release. Pin Python 3.12.9 from ~/scratch/miniconda3.

## Skipped Steps (need human attention)
- (none yet)
