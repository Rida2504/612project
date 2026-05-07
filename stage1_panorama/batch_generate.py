"""
Stage 1 Batch Generator

Runs all scene configs in configs/prompts/ × N seeds and outputs:
  - One PNG per (scene, seed) in --out-dir
  - batch_results.csv with CLIP scores + QA metrics

Usage:
    python batch_generate.py --out-dir ~/textworld/corpus_output --seeds 42,7,13
    python batch_generate.py --out-dir outputs/panoramas --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
import yaml


CONFIGS_DIR = Path(__file__).parent / "configs" / "prompts"
DEFAULT_SEEDS = [42, 7, 13]


def load_scene_configs(configs_dir: Path) -> list[dict]:
    scenes = []
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        cfg["slug"] = yaml_path.stem
        scenes.append(cfg)
    return scenes


def run_batch(
    out_dir: str,
    seeds: list[int],
    device: str = "cuda",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
) -> list[dict]:
    from stage1_panorama.generate_panorama import load_pipeline, generate
    from stage1_panorama.panorama_qa import load_clip, qa_image

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenes = load_scene_configs(CONFIGS_DIR)
    if not scenes:
        raise RuntimeError(f"No YAML configs found in {CONFIGS_DIR}")

    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading pipeline on {device}...")
    pipe = load_pipeline(device, dtype)
    clip_state = load_clip(device)

    total = len(scenes) * len(seeds)
    print(f"\nGenerating {total} panoramas ({len(scenes)} scenes × {len(seeds)} seeds)\n")

    results = []
    idx = 0
    t_start = time.time()

    for scene in scenes:
        slug = scene["slug"]
        prompt = scene["prompt"]

        for seed in seeds:
            idx += 1
            fname = f"{slug}_s{seed}.png"
            img_path = str(out_path / fname)

            print(f"[{idx}/{total}] {slug} seed={seed}")
            t0 = time.time()

            try:
                generate(
                    prompt=prompt,
                    seed=seed,
                    out_path=img_path,
                    device=device,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    pipe=pipe,
                )
                elapsed = round(time.time() - t0, 1)

                qa = qa_image(img_path, prompt, clip_state, device)
                qa["slug"] = slug
                qa["seed"] = seed
                qa["gen_time_s"] = elapsed
                results.append(qa)

                status = "PASS" if qa["qa_pass"] else "FAIL"
                print(
                    f"  [{status}] CLIP={qa['clip_score']:.4f} "
                    f"pole={qa['pole_variance']} seam={qa['seam_diff']} ({elapsed}s)"
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "path": img_path,
                    "prompt": prompt,
                    "slug": slug,
                    "seed": seed,
                    "clip_score": 0.0,
                    "clip_pass": False,
                    "pole_variance": 0.0,
                    "pole_pass": False,
                    "seam_diff": 0.0,
                    "seam_pass": False,
                    "qa_pass": False,
                    "gen_time_s": round(time.time() - t0, 1),
                    "error": str(e),
                })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    total_time = round(time.time() - t_start, 1)

    csv_path = out_path / "batch_results.csv"
    if results:
        fieldnames = [k for k in results[0].keys()]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    n_pass = sum(1 for r in results if r.get("qa_pass"))
    clip_scores = [r["clip_score"] for r in results if r["clip_score"] > 0]
    avg_clip = round(sum(clip_scores) / len(clip_scores), 4) if clip_scores else 0.0
    min_clip = round(min(clip_scores), 4) if clip_scores else 0.0
    max_clip = round(max(clip_scores), 4) if clip_scores else 0.0

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Generated : {len(results)}/{total}")
    print(f"  QA passed : {n_pass}/{len(results)}")
    print(f"  CLIP avg  : {avg_clip}  min={min_clip}  max={max_clip}")
    print(f"  Total time: {total_time}s ({total_time/60:.1f} min)")
    print(f"  Results   : {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 1 batch panorama generator")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path.home() / "textworld" / "corpus_output"),
    )
    parser.add_argument("--seeds", type=str, default="42,7,13")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=7.5)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    run_batch(
        out_dir=args.out_dir,
        seeds=seeds,
        device=args.device,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )


if __name__ == "__main__":
    main()
