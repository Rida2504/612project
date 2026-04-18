"""
TextWorld VR: Batch Scene Generator

Generates multiple scenes end-to-end for benchmarking and demo.
Runs Stages 1-3 for each prompt, collects metrics, and produces a summary.

Usage:
    python batch_generate.py                          # use default scenes
    python batch_generate.py --prompts-file scenes.txt  # custom scene list
    python batch_generate.py --stages 1 2             # only panorama + views
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import yaml

# Default scene prompts (indoor/room-scale focus for the project)
DEFAULT_PROMPTS = [
    "a cozy Japanese coffee shop",
    "a medieval castle great hall with stone walls and chandeliers",
    "a modern minimalist living room with large windows",
    "a colorful kindergarten classroom with toys",
    "a cyberpunk neon-lit bar with holographic displays",
    "a rustic Italian kitchen with brick oven",
    "a luxurious hotel lobby with marble floors",
    "a cozy library with floor to ceiling bookshelves",
    "a futuristic space station control room",
    "a tropical beach resort lounge with ocean view",
    "a vintage 1950s American diner",
    "an ancient Egyptian temple with hieroglyphics",
]


def run_batch(
    prompts: list[str],
    config_path: str = "configs/default.yaml",
    stages: list[int] = [1, 2, 3],
    seed: int = 42,
    use_v2_trainer: bool = True,
):
    """Run the pipeline for each prompt and collect results."""
    from run_pipeline import run_stage1, run_stage2

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = []
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"TextWorld VR: Batch Generation")
    print(f"{'='*60}")
    print(f"Scenes: {len(prompts)}")
    print(f"Stages: {stages}")
    print(f"Trainer: {'v2 (enhanced)' if use_v2_trainer else 'v1 (basic)'}")
    print(f"{'='*60}\n")

    for i, prompt in enumerate(prompts):
        scene_start = time.time()
        print(f"\n{'─'*60}")
        print(f"Scene {i+1}/{len(prompts)}: {prompt}")
        print(f"{'─'*60}")

        scene_result = {
            "prompt": prompt,
            "seed": seed,
            "index": i,
        }

        try:
            panorama_path = None
            multiview_dir = None
            splat_path = None

            # Stage 1: Panorama
            if 1 in stages:
                t0 = time.time()
                panorama_path = run_stage1(prompt, config, seed, output_path=None)
                scene_result["panorama"] = panorama_path
                scene_result["stage1_time"] = round(time.time() - t0, 1)
                print(f"  Stage 1: {scene_result['stage1_time']}s")

            # Stage 2: Multi-view
            if 2 in stages:
                if panorama_path is None:
                    # Try to find existing panorama
                    safe_name = prompt[:50].replace(" ", "_").replace("/", "-")
                    pano_path = Path(config.get("output_dir", "outputs")) / "panoramas" / f"{safe_name}_s{seed}.png"
                    if pano_path.exists():
                        panorama_path = str(pano_path)

                if panorama_path:
                    t0 = time.time()
                    multiview_dir = run_stage2(panorama_path, config)
                    scene_result["multiview_dir"] = multiview_dir
                    scene_result["stage2_time"] = round(time.time() - t0, 1)
                    print(f"  Stage 2: {scene_result['stage2_time']}s")

            # Stage 3: 3DGS
            if 3 in stages:
                if multiview_dir is None:
                    safe_name = prompt[:50].replace(" ", "_").replace("/", "-")
                    mv_dir = Path(config.get("output_dir", "outputs")) / "multiview" / f"{safe_name}_s{seed}"
                    if mv_dir.exists():
                        multiview_dir = str(mv_dir)

                if multiview_dir:
                    t0 = time.time()
                    output_dir = Path(config.get("output_dir", "outputs")) / "splats"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    scene_name = Path(multiview_dir).name
                    splat_path = str(output_dir / f"{scene_name}.ply")

                    if use_v2_trainer:
                        from stage3_3dgs.train_3dgs_v2 import train as train_v2
                        train_v2(
                            multiview_dir,
                            splat_path,
                            num_iters=3000,
                            num_points=10000,
                            downscale=2,
                            device_str=config.get("panorama", {}).get("device", "mps"),
                            save_interval=1000,
                        )
                    else:
                        from stage3_3dgs.train_3dgs import train as train_v1
                        train_v1(
                            multiview_dir,
                            splat_path,
                            num_iters=2000,
                            num_points=5000,
                            downscale=2,
                            device_str=config.get("panorama", {}).get("device", "mps"),
                        )

                    scene_result["splat"] = splat_path
                    scene_result["stage3_time"] = round(time.time() - t0, 1)
                    print(f"  Stage 3: {scene_result['stage3_time']}s")

            scene_result["total_time"] = round(time.time() - scene_start, 1)
            scene_result["status"] = "success"

        except Exception as e:
            scene_result["status"] = "error"
            scene_result["error"] = str(e)
            print(f"  ERROR: {e}")

        results.append(scene_result)

        # Free GPU memory between scenes
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    n_success = sum(1 for r in results if r["status"] == "success")
    print(f"  Completed: {n_success}/{len(prompts)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if n_success > 0:
        times = [r.get("total_time", 0) for r in results if r["status"] == "success"]
        print(f"  Avg time/scene: {sum(times)/len(times):.1f}s")

    # Save results
    output_path = Path(config.get("output_dir", "outputs")) / "batch_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "total_time_s": round(total_time, 1),
            "num_scenes": len(prompts),
            "num_success": n_success,
            "stages": stages,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TextWorld VR Batch Generator")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Text file with one prompt per line")
    parser.add_argument("--prompts", nargs="+", type=str, default=None,
                        help="Scene prompts (space-separated)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-scenes", type=int, default=None,
                        help="Limit number of scenes (default: all)")
    parser.add_argument("--v1", action="store_true", help="Use v1 trainer instead of v2")
    args = parser.parse_args()

    # Determine prompts
    if args.prompts:
        prompts = args.prompts
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        prompts = DEFAULT_PROMPTS

    if args.num_scenes:
        prompts = prompts[:args.num_scenes]

    run_batch(
        prompts,
        config_path=args.config,
        stages=args.stages,
        seed=args.seed,
        use_v2_trainer=not args.v1,
    )


if __name__ == "__main__":
    main()
