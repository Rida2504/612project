"""
TextWorld VR: End-to-End Pipeline Runner
Text → 360° Panorama → Multi-View Extraction → 3DGS → VR Export

Usage:
    python run_pipeline.py "a cozy Japanese coffee shop"
    python run_pipeline.py "a cozy Japanese coffee shop" --stages 1 2 3 --seed 42
    python run_pipeline.py "scene" --stages 3 4 --multiview-dir outputs/multiview/scene
    python run_pipeline.py "scene" --stages 4 --splat outputs/splats/scene.ply --unity-project ~/Unity/TextWorldVR
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from pathlib import Path
from typing import Optional

import yaml


def run_stage1(prompt: str, config: dict, seed: int | None, output_path: str | None) -> str:
    """Stage 1: Generate 360° panorama from text."""
    from stage1_panorama.generate import load_pipeline, build_prompt, generate_panorama

    cfg = config["panorama"]
    pipe = load_pipeline(cfg)
    full_prompt = build_prompt(prompt, cfg)
    image = generate_panorama(pipe, full_prompt, cfg, seed=seed)

    # Save panorama
    output_dir = Path(config.get("output_dir", "outputs")) / "panoramas"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        out = Path(output_path)
    else:
        safe_name = prompt[:50].replace(" ", "_").replace("/", "-")
        suffix = f"_s{seed}" if seed is not None else ""
        out = output_dir / f"{safe_name}{suffix}.png"

    image.save(out, "PNG")
    print(f"[Stage 1] Panorama saved: {out}")

    # Cleanup GPU memory
    del pipe
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(out)


def run_stage2(panorama_path: str, config: dict) -> str:
    """Stage 2: Extract multi-view images from panorama."""
    from stage2_multiview.extract_views import extract_multiviews

    mv_cfg = config["multiview"]
    pano_name = Path(panorama_path).stem
    output_dir = str(Path(config.get("output_dir", "outputs")) / "multiview" / pano_name)

    extract_multiviews(
        panorama_path,
        output_dir,
        num_views=mv_cfg["num_views"],
        fov_deg=mv_cfg["fov_degrees"],
        out_w=mv_cfg["output_width"],
        out_h=mv_cfg["output_height"],
        elevation_angles=mv_cfg.get("elevation_angles", [0.0]),
    )
    print(f"[Stage 2] Multi-views saved: {output_dir}")
    return output_dir


def run_stage3(multiview_dir: str, config: dict, use_v2: bool = True) -> str:
    """Stage 3: Run 3DGS reconstruction."""
    output_dir = Path(config.get("output_dir", "outputs")) / "splats"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_name = Path(multiview_dir).name
    device_str = config.get("panorama", {}).get("device", "mps")

    gs_cfg = config.get("gaussian_splatting", {})

    if use_v2:
        from stage3_3dgs.train_3dgs_v2 import train as train_v2
        output_ply = str(output_dir / f"{scene_name}.ply")
        train_v2(
            multiview_dir,
            output_ply,
            num_iters=gs_cfg.get("num_iters", 3000),
            num_points=gs_cfg.get("num_points", 10000),
            lr=gs_cfg.get("lr", 0.005),
            downscale=gs_cfg.get("downscale", 2),
            device_str=device_str,
            sh_degree=gs_cfg.get("sh_degree", 3),
            max_gaussians=gs_cfg.get("max_gaussians", 50000),
            lambda_ssim=gs_cfg.get("lambda_ssim", 0.2),
            save_interval=gs_cfg.get("save_interval", 500),
        )
    else:
        from stage3_3dgs.train_3dgs import train as train_v1
        output_ply = str(output_dir / f"{scene_name}.ply")
        train_v1(
            multiview_dir,
            output_ply,
            num_iters=gs_cfg.get("num_iters", 2000),
            num_points=gs_cfg.get("num_points", 5000),
            downscale=gs_cfg.get("downscale", 2),
            device_str=device_str,
        )

    print(f"[Stage 3] Splat saved: {output_ply}")
    return output_ply


def run_stage4(splat_ply: str, config: dict, unity_project: str | None = None) -> str:
    """Stage 4: Export for VR rendering."""
    from stage4_vr.export_for_unity import prepare_unity_assets

    if unity_project is None:
        unity_project = config.get("vr", {}).get("unity_project", "~/Unity/TextWorldVR")
        unity_project = str(Path(unity_project).expanduser())

    scene_name = Path(splat_ply).stem
    prepare_unity_assets(splat_ply, unity_project, scene_name)
    print(f"[Stage 4] Unity assets exported to: {unity_project}")
    return unity_project


def run_evaluate(prompt: str, panorama_path: str, multiview_dir: str, splat_ply: str, config: dict):
    """Run evaluation metrics on the generated scene."""
    from evaluate import compute_clip_score, evaluate_render_quality
    import json

    print("\n[Evaluate] Computing metrics...")
    results = {"prompt": prompt}

    # CLIP score
    try:
        clip = compute_clip_score(prompt, panorama_path)
        results["clip_score"] = round(clip, 4)
        print(f"  CLIP Score: {clip:.4f}")
    except Exception as e:
        print(f"  CLIP Score: skipped ({e})")

    # Render quality
    try:
        render_metrics = evaluate_render_quality(multiview_dir, splat_ply)
        results.update(render_metrics)
    except Exception as e:
        print(f"  Render metrics: skipped ({e})")

    # Save
    eval_path = Path(splat_ply).with_suffix(".eval.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Evaluate] Saved: {eval_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="TextWorld VR: Text → 360° Panorama → Multi-Views → 3DGS → VR"
    )
    parser.add_argument("prompt", type=str, help="Text description of the scene")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3],
                        help="Which stages to run (default: 1 2 3)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--panorama", type=str, default=None,
                        help="Skip stage 1: use existing panorama path")
    parser.add_argument("--multiview-dir", type=str, default=None,
                        help="Skip stages 1-2: use existing multi-view directory")
    parser.add_argument("--splat", type=str, default=None,
                        help="Skip stages 1-3: use existing .ply splat")
    parser.add_argument("--unity-project", type=str, default=None,
                        help="Unity project path for Stage 4")
    parser.add_argument("--v1", action="store_true",
                        help="Use v1 3DGS trainer (simpler, faster)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation metrics after pipeline")
    parser.add_argument("--viewer", action="store_true",
                        help="Open web viewer after pipeline")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    t_start = time.time()
    panorama_path = args.panorama
    multiview_dir = args.multiview_dir
    splat_ply = args.splat

    # Stage 1
    if 1 in args.stages and panorama_path is None and multiview_dir is None and splat_ply is None:
        print("\n" + "=" * 60)
        print("STAGE 1: Text → 360° Panorama")
        print("=" * 60)
        panorama_path = run_stage1(args.prompt, config, args.seed, output_path=None)

    # Stage 2
    if 2 in args.stages and multiview_dir is None and splat_ply is None:
        if panorama_path is None:
            print("ERROR: Need a panorama for Stage 2. Run Stage 1 or pass --panorama.")
            return
        print("\n" + "=" * 60)
        print("STAGE 2: Panorama → Multi-View Extraction")
        print("=" * 60)
        multiview_dir = run_stage2(panorama_path, config)

    # Stage 3
    if 3 in args.stages and splat_ply is None:
        if multiview_dir is None:
            print("ERROR: Need multi-view dir for Stage 3. Run Stages 1-2 or pass --multiview-dir.")
            return
        print("\n" + "=" * 60)
        print("STAGE 3: Multi-Views → 3D Gaussian Splatting")
        print("=" * 60)
        splat_ply = run_stage3(multiview_dir, config, use_v2=not args.v1)

    # Stage 4
    if 4 in args.stages:
        if splat_ply is None:
            print("ERROR: Need .ply splat for Stage 4. Run Stages 1-3 or pass --splat.")
            return
        print("\n" + "=" * 60)
        print("STAGE 4: 3DGS → VR Export (Unity)")
        print("=" * 60)
        run_stage4(splat_ply, config, args.unity_project)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Print summary of outputs
    if panorama_path:
        print(f"  Panorama:   {panorama_path}")
    if multiview_dir:
        print(f"  Multi-view: {multiview_dir}")
    if splat_ply:
        print(f"  Splat:      {splat_ply}")

    # Evaluate
    if args.evaluate and panorama_path and multiview_dir and splat_ply:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        run_evaluate(args.prompt, panorama_path, multiview_dir, splat_ply, config)

    # Open viewer
    if args.viewer:
        import webbrowser
        viewer_path = Path(__file__).parent / "viewer" / "index.html"
        print(f"\nOpening viewer: {viewer_path}")
        webbrowser.open(f"file://{viewer_path}")


if __name__ == "__main__":
    main()
