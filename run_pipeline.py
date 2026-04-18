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


def run_stage2_depth(panorama_path: str, config: dict) -> str:
    """Stage 2 (new): panorama → monocular depth → colored point cloud (.ply).

    This replaces the parallax-less `extract_views` approach of the legacy
    Stage 2 with a real 3D point cloud suitable for 3DGS initialization.
    """
    from stage2_multiview.pano_depth import pano_to_pointcloud

    depth_cfg = config.get("depth", {}) or {}
    pano_name = Path(panorama_path).stem
    output_dir = Path(config.get("output_dir", "outputs")) / "pointclouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_ply = str(output_dir / f"{pano_name}.ply")

    pano_to_pointcloud(
        panorama_path,
        out_ply,
        depth_model=depth_cfg.get("model", "depth-anything/Depth-Anything-V2-Small-hf"),
        max_points=depth_cfg.get("max_points", 500_000),
        fallback_if_no_weights=depth_cfg.get("fallback_if_no_weights", False),
        use_model=depth_cfg.get("use_model", True),
        seed=depth_cfg.get("seed", 0),
        save_depth_viz=depth_cfg.get("save_depth_viz", True),
    )
    print(f"[Stage 2-depth] Point cloud saved: {out_ply}")
    return out_ply


def run_stage3_gsplat(multiview_dir: str, init_ply: str, config: dict) -> str:
    """Stage 3 (new): gsplat-based 3DGS training with depth-initialized Gaussians."""
    from stage3_3dgs.train_gsplat import train_gsplat

    gs_cfg = config.get("gaussian_splatting", {}) or {}
    output_dir = Path(config.get("output_dir", "outputs")) / "splats"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_name = Path(multiview_dir).name
    out_ply = str(output_dir / f"{scene_name}_gsplat.ply")
    log_csv = str(output_dir / f"{scene_name}_gsplat.log.csv")
    device_str = config.get("panorama", {}).get("device", "mps")

    train_gsplat(
        multiview_dir,
        init_ply,
        out_ply,
        num_iters=gs_cfg.get("num_iters", 500),
        device_str=device_str,
        downscale=gs_cfg.get("downscale", 2),
        sh_degree=gs_cfg.get("sh_degree", 1),
        max_gaussians=gs_cfg.get("max_gaussians", 100_000),
        log_csv=log_csv,
        force_fallback=gs_cfg.get("force_fallback", False),
    )
    print(f"[Stage 3-gsplat] Splat saved: {out_ply}")
    return out_ply


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
    parser.add_argument("--use-depth-init", action="store_true", default=True,
                        help="New default: Stage 2 = panorama depth → point cloud, "
                             "Stage 3 = gsplat with depth-based Gaussian init. Addresses "
                             "the parallax-less-multiview issue of the legacy pipeline.")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the legacy extract_views + random-init v2 trainer path.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation metrics after pipeline")
    parser.add_argument("--viewer", action="store_true",
                        help="Open web viewer after pipeline")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "mps", "cpu"],
                        help="Override config device for all stages (cluster: 'cuda').")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override config output_dir (e.g., BeeOND workdir).")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override gaussian_splatting.num_iters.")
    parser.add_argument("--max-gaussians", type=int, default=None,
                        help="Override gaussian_splatting.max_gaussians.")
    args = parser.parse_args()

    # --legacy overrides --use-depth-init
    if args.legacy:
        args.use_depth_init = False

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.device:
        config.setdefault("panorama", {})["device"] = args.device
        config.setdefault("depth", {})["device"] = args.device
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.iters is not None:
        config.setdefault("gaussian_splatting", {})["num_iters"] = args.iters
    if args.max_gaussians is not None:
        config.setdefault("gaussian_splatting", {})["max_gaussians"] = args.max_gaussians

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

    # Stage 2 — when --use-depth-init (default), run BOTH:
    #   (a) pano → depth → point cloud (for 3DGS init)
    #   (b) extract perspective views (for photometric supervision)
    init_ply = None
    if 2 in args.stages and multiview_dir is None and splat_ply is None:
        if panorama_path is None:
            print("ERROR: Need a panorama for Stage 2. Run Stage 1 or pass --panorama.")
            return
        if args.use_depth_init:
            print("\n" + "=" * 60)
            print("STAGE 2a: Panorama → Depth → Point Cloud (new, depth-init)")
            print("=" * 60)
            init_ply = run_stage2_depth(panorama_path, config)
        print("\n" + "=" * 60)
        print("STAGE 2b: Panorama → Multi-View Extraction (for supervision)")
        print("=" * 60)
        multiview_dir = run_stage2(panorama_path, config)

    # Stage 3 — route to gsplat (depth-init) or legacy v2 trainer
    if 3 in args.stages and splat_ply is None:
        if multiview_dir is None:
            print("ERROR: Need multi-view dir for Stage 3. Run Stages 1-2 or pass --multiview-dir.")
            return
        if args.use_depth_init:
            if init_ply is None:
                # Try to locate an existing pointcloud .ply
                pano_stem = Path(panorama_path).stem if panorama_path else Path(multiview_dir).name
                guess = Path(config.get("output_dir", "outputs")) / "pointclouds" / f"{pano_stem}.ply"
                if guess.exists():
                    init_ply = str(guess)
                else:
                    print(f"ERROR: --use-depth-init requires a point cloud. Run stage 2 or "
                          f"provide --panorama. Expected: {guess}")
                    return
            print("\n" + "=" * 60)
            print("STAGE 3: gsplat (CUDA) with depth-initialized Gaussians")
            print("=" * 60)
            splat_ply = run_stage3_gsplat(multiview_dir, init_ply, config)
        else:
            print("\n" + "=" * 60)
            print("STAGE 3: Multi-Views → 3D Gaussian Splatting (LEGACY)")
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
