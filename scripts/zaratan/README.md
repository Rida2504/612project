# Zaratan workspace (UMD HPC)

Cross-node workspace design for 2 × H100 single-node jobs on the `mqzhu-prj-aac`
allocation. Goal: each node has a **fast 12 TB /tmp working directory** with a
periodic rsync back to `~/scratch/phase4/textworld-vr/` (BeeGFS, durable,
cross-node visible) so both nodes agree on a source of truth.

## Storage tiers

| Tier | Path | Size | Speed | Durability |
|---|---|---|---|---|
| Home | `~/` | 10 GB | slow | backed up |
| **Source of truth** | `~/scratch/phase4/textworld-vr/` (= `/scratch/zt1/project/mqzhu-prj/user/yog/phase4/textworld-vr/`) | 7 TB free | BeeGFS, both-nodes visible | durable |
| **Per-node hot** | `/tmp/$USER/textworld-vr/` (on gpu-a6-X) | 12 TB NVMe | fast, node-local | wiped at job end |
| **Cross-node view** | `/tmp/$USER/peer-<other-node>/` | via sshfs | ~100 MB/s, POSIX-lite | node-lifetime |

## The three active directories on each node

```
/tmp/$USER/textworld-vr/
├── code/          ← mirror of ~/scratch/.../code/ (rsync-in at start)
├── outputs/       ← job writes here; rsync-out periodically
├── hf_cache/      ← HF_HOME (models downloaded once per session)
├── venv/          ← symlink to a shared venv in ~/scratch/.../shared/venv/
└── peer-<host>/   ← sshfs of the OTHER node's /tmp/$USER/textworld-vr/
```

## Scripts

| Script | When to run | What it does |
|---|---|---|
| `bootstrap_node.sh` | On `srun` / `ssh` into a node, once per session | mkdirs, pulls code from `~/scratch`, sets env (HF_HOME, PYTHONPATH) |
| `mount_peer.sh <other-hostname>` | After bootstrap, if cross-node view needed | sshfs the peer's `/tmp/$USER/textworld-vr/` under local `/tmp/$USER/textworld-vr/peer-<host>/` |
| `sync_out.sh` | Background daemon (nohup or tmux) | rsync `/tmp/.../outputs/` → `~/scratch/.../outputs/` every 60s |
| `umount_peer.sh` | End of session | `fusermount -u` the peer mount |
| `run_pipeline.sh <prompt> [seed]` | Per scene | wraps `run_pipeline.py` with the correct env and output paths |

## Submission

Two jobs, both on `mqzhu-prj-aac`, both 1 node exclusive 24h. Submitted
independently so the current `persist-mqzhu` (job 19035335 on gpu-a6-9) can
keep running; only the `persist-msml605` job (19035334 on gpu-a6-8) is
replaced.

```bash
# Replace the msml605 job with a mqzhu-prj-aac 24h exclusive job:
scancel 19035334
sbatch --parsable scripts/zaratan/sbatch_persist.sbatch
```

See `sbatch_persist.sbatch` for the full SLURM spec.

## Why this design (vs native `/scratch/local/$SLURM_JOB_ID` striping)

Native striping requires ONE `--nodes=2 --exclusive` job, which charges to ONE
account. This setup preserves two separate single-node jobs (both on
`mqzhu-prj-aac` since `msml605-class` is already over quota) and approximates
striped behavior via:
- shared source-of-truth on BeeGFS (`~/scratch`)
- fast per-node working copy on `/tmp`
- optional sshfs for cross-node reads
- periodic rsync from `/tmp` to `~/scratch` for durability

Trade-off: write-collisions between the two nodes on the same file are
possible if both touch the same output. We sidestep this by partitioning the
scene corpus — each node processes a disjoint set of prompts.
