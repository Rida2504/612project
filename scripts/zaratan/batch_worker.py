"""Per-GPU worker: process a list of (prompt, seed) tasks in ONE process.

Avoids paying the ~60s CUDA-context init + gsplat JIT load once per task.
Instead initializes CUDA + imports heavy modules once, then iterates over tasks.

Reads tasks from stdin as "<prompt>\t<seed>" lines.
Writes one-line status per task to stdout (also teed into a main log by the shell).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


def _expected_splat(prompt: str, seed: int, config: dict) -> Path:
    """Mirror run_pipeline's output naming for the final splat .ply."""
    safe_name = prompt[:50].replace(" ", "_").replace("/", "-")
    suffix = f"_s{seed}"
    return Path(config["output_dir"]) / "splats" / f"{safe_name}{suffix}_gsplat.ply"


def run_one(prompt: str, seed: int, config: dict, skip_if_exists: bool = True) -> dict:
    """Run a single pipeline invocation. Returns a status dict."""
    from run_pipeline import run_stage1, run_stage2, run_stage2_depth, run_stage3_gsplat

    result = {"prompt": prompt, "seed": seed, "status": "ok", "error": None,
              "panorama": None, "multiview": None, "splat": None,
              "elapsed": None}
    exp = _expected_splat(prompt, seed, config)
    if skip_if_exists and exp.exists() and exp.stat().st_size > 1_000_000:
        result["status"] = "skip-exists"
        result["splat"] = str(exp)
        result["elapsed"] = 0.0
        return result
    t0 = time.time()
    try:
        pano = run_stage1(prompt, config, seed, output_path=None)
        result["panorama"] = pano
        init_ply = run_stage2_depth(pano, config)
        multiview = run_stage2(pano, config)
        result["multiview"] = multiview
        splat = run_stage3_gsplat(multiview, init_ply, config)
        result["splat"] = splat
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    result["elapsed"] = round(time.time() - t0, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--max-gaussians", type=int, default=300_000)
    parser.add_argument("--tag", type=str, default="",
                        help="Label prefix for stdout status lines.")
    parser.add_argument("--rerun", action="store_true",
                        help="Redo tasks even if the expected splat already exists.")
    args = parser.parse_args()

    # Eager heavy imports (pay cost once)
    import yaml
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[2] / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides (mirrors run_pipeline.py)
    config.setdefault("panorama", {})["device"] = args.device
    config.setdefault("depth", {})["device"] = args.device
    config["output_dir"] = args.output_dir
    config.setdefault("gaussian_splatting", {})["num_iters"] = args.iters
    config.setdefault("gaussian_splatting", {})["max_gaussians"] = args.max_gaussians

    tag = args.tag or os.environ.get("BATCH_TAG", "worker")

    # Warm CUDA + torch on first line — forces context init BEFORE first task's timing
    t_init = time.time()
    import torch
    _ = torch.zeros(1, device="cuda") if args.device == "cuda" and torch.cuda.is_available() else None
    import gsplat  # noqa: F401
    print(f"[{tag}] warm: cuda+torch+gsplat imported in {time.time()-t_init:.1f}s", flush=True)

    # Drain stdin for tasks
    tasks = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"[{tag}] skipping bad line: {line!r}", flush=True)
            continue
        tasks.append((parts[0], int(parts[1])))

    print(f"[{tag}] got {len(tasks)} tasks", flush=True)

    ok = 0
    t_total0 = time.time()
    for i, (prompt, seed) in enumerate(tasks):
        print(f"[{tag}] {i+1}/{len(tasks)} START prompt={prompt!r} seed={seed}", flush=True)
        r = run_one(prompt, seed, config, skip_if_exists=not args.rerun)
        status = r["status"]
        print(f"[{tag}] {i+1}/{len(tasks)} END   status={status} elapsed={r['elapsed']}s "
              f"splat={r.get('splat')} err={r.get('error')}", flush=True)
        if status == "ok":
            ok += 1
    print(f"[{tag}] DONE ok={ok}/{len(tasks)} total={time.time()-t_total0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
