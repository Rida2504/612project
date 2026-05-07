"""
Stage 2 (replacement): Equirectangular Panorama → Monocular Depth → Point Cloud

Fixes the parallax-less reprojection problem of the old `extract_views.py` by
producing a genuine 3D point cloud that can initialize 3DGS.

Pipeline:
  panorama (H, W, 3) → depth (H, W) → 3D points via equirect ray-casting + depth scale
  → random subsample → .ply (XYZ + RGB)

Depth backend:
  - Default: Depth Anything v2 via HuggingFace transformers (`depth-anything/Depth-Anything-V2-Small-hf`)
  - Fallback: unit-sphere (radius=1) — degenerate but keeps the pipeline runnable
    without model weights. Emits a loud warning.

Usage:
  python stage2_multiview/pano_depth.py outputs/panoramas/scene.png \\
      --output outputs/pointclouds/scene.ply [--max-points 500000] [--fallback-if-no-weights]
"""

from __future__ import annotations

import os
<<<<<<< HEAD
=======

>>>>>>> main
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("ERROR: plyfile not installed. Run: pip install plyfile", file=sys.stderr)
    raise


# ─── Depth estimation ────────────────────────────────────────────────────────

<<<<<<< HEAD
def _pick_device() -> str:
    try:
        import torch
=======

def _pick_device() -> str:
    try:
        import torch

>>>>>>> main
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _unit_sphere_depth(H: int, W: int) -> np.ndarray:
    """Degenerate depth: every pixel at unit distance. Preserves plumbing when no model available."""
    return np.ones((H, W), dtype=np.float32)


def estimate_pano_depth(
    pano_path: str,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    fallback_if_no_weights: bool = False,
    use_model: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Run monocular depth estimation on an equirectangular panorama.

    Note: perspective Depth Anything v2 is known to warp near the poles of an
    equirectangular image. For production, swap to a panorama-aware model
    (DA360/PanDA/DAP) — see findings doc.

    Args:
        use_model: if False, skip model entirely and return unit-sphere depth.
            Useful for pipeline smoke tests and environments where the ML stack
            is unstable (e.g. protobuf conflicts on Mac). Does NOT require
            `--fallback-if-no-weights` — forces the fallback unconditionally.
        fallback_if_no_weights: if True AND model loading/inference raises a
            catchable Python exception, fall back to unit-sphere depth.
            Cannot catch native C++ crashes; prefer `use_model=False` in that case.

    Returns: (H, W) float32 relative depth map; values are NOT metric.
    """
    device = device or _pick_device()
    pano = np.array(Image.open(pano_path).convert("RGB"))
    H, W = pano.shape[:2]

    if not use_model:
        print("[pano_depth] use_model=False → unit-sphere depth (plumbing check only)")
        return _unit_sphere_depth(H, W)

    try:
        from transformers import pipeline
    except ImportError:
        if fallback_if_no_weights:
<<<<<<< HEAD
            warnings.warn("transformers not installed — falling back to unit-sphere depth")
=======
            warnings.warn(
                "transformers not installed — falling back to unit-sphere depth"
            )
>>>>>>> main
            return _unit_sphere_depth(H, W)
        raise

    try:
        print(f"[pano_depth] Loading {model_name} on {device} ...")
        # Force PyTorch backend; transformers can otherwise pick TF and crash on Mac
        depth_pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=device,
            framework="pt",
        )
    except Exception as e:
        if fallback_if_no_weights:
<<<<<<< HEAD
            warnings.warn(f"Depth model load failed ({e}); falling back to unit-sphere depth")
=======
            warnings.warn(
                f"Depth model load failed ({e}); falling back to unit-sphere depth"
            )
>>>>>>> main
            return np.ones((H, W), dtype=np.float32)
        raise

    print(f"[pano_depth] Running depth on {W}x{H} panorama ...")
    img = Image.fromarray(pano)
    try:
        out = depth_pipe(img)
    except Exception as e:
        if fallback_if_no_weights:
<<<<<<< HEAD
            warnings.warn(f"Depth inference failed ({e}); falling back to unit-sphere depth")
=======
            warnings.warn(
                f"Depth inference failed ({e}); falling back to unit-sphere depth"
            )
>>>>>>> main
            return np.ones((H, W), dtype=np.float32)
        raise
    depth = np.array(out["depth"], dtype=np.float32)

    # Normalize: the HF pipeline returns disparity-like values; invert + normalize
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    # Map [0,1] disparity-like to [0.3, 3.0] pseudo-metric (indoor scale)
    # Closer = smaller depth, so invert: large disparity → small depth
    # Standard: raw is disparity, so depth = 1 / (disparity + eps)
    depth_inv = 1.0 / (depth + 0.05)
    depth_inv = depth_inv / np.median(depth_inv)  # normalize to unit median
    depth_m = np.clip(depth_inv * 1.5, 0.3, 6.0).astype(np.float32)  # meters, clamped

    return depth_m


# ─── Equirectangular pixel → world ray ────────────────────────────────────────

<<<<<<< HEAD
=======

>>>>>>> main
def equirect_rays(H: int, W: int) -> np.ndarray:
    """
    Per-pixel unit ray vectors for an equirectangular image.
    Returns: (H, W, 3) where each pixel (v, u) → (x, y, z) unit vector.

    Convention: +x right, +y up, -z forward (OpenGL). Panorama is oriented so
    u=W/2 points toward +z (front), pitch=0 at horizon (v=H/2).
    """
<<<<<<< HEAD
    u = (np.arange(W, dtype=np.float32) + 0.5) / W   # [0,1]
    v = (np.arange(H, dtype=np.float32) + 0.5) / H   # [0,1]
    theta = (u * 2.0 - 1.0) * math.pi                # longitude  [-π, π]
    phi = (0.5 - v) * math.pi                        # latitude   [π/2, -π/2]

    U, V = np.meshgrid(theta, phi)                   # (H, W)
=======
    u = (np.arange(W, dtype=np.float32) + 0.5) / W  # [0,1]
    v = (np.arange(H, dtype=np.float32) + 0.5) / H  # [0,1]
    theta = (u * 2.0 - 1.0) * math.pi  # longitude  [-π, π]
    phi = (0.5 - v) * math.pi  # latitude   [π/2, -π/2]

    U, V = np.meshgrid(theta, phi)  # (H, W)
>>>>>>> main
    cos_phi = np.cos(V)
    x = np.sin(U) * cos_phi
    y = np.sin(V)
    z = np.cos(U) * cos_phi
    rays = np.stack([x, y, -z], axis=-1).astype(np.float32)  # flip z: pano 'front' → -z
    return rays


# ─── Point cloud construction ─────────────────────────────────────────────────

<<<<<<< HEAD
=======

>>>>>>> main
def pano_to_pointcloud(
    pano_path: str,
    out_ply_path: str,
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
    max_points: int = 500_000,
    fallback_if_no_weights: bool = False,
    use_model: bool = True,
    seed: int = 0,
    save_depth_viz: bool = True,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Convert an equirectangular panorama to a colored 3D point cloud.

    Returns: (ply_path, points [N,3], colors [N,3] uint8)
    """
    pano = np.array(Image.open(pano_path).convert("RGB"))
    H, W = pano.shape[:2]

    depth = estimate_pano_depth(
<<<<<<< HEAD
        pano_path, model_name=depth_model,
=======
        pano_path,
        model_name=depth_model,
>>>>>>> main
        fallback_if_no_weights=fallback_if_no_weights,
        use_model=use_model,
    )
    if depth.shape != (H, W):
        # Resize depth to panorama size if needed (some HF pipelines return a smaller map)
        depth_img = Image.fromarray(depth).resize((W, H), Image.BILINEAR)
        depth = np.array(depth_img, dtype=np.float32)

<<<<<<< HEAD
    rays = equirect_rays(H, W)                   # (H, W, 3)
    points = rays * depth[..., None]             # (H, W, 3)
=======
    rays = equirect_rays(H, W)  # (H, W, 3)
    points = rays * depth[..., None]  # (H, W, 3)
>>>>>>> main
    points = points.reshape(-1, 3)
    colors = pano.reshape(-1, 3).astype(np.uint8)

    # Subsample
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    out_path = Path(out_ply_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")]
=======
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
>>>>>>> main
    verts = np.empty(points.shape[0], dtype=vertex_dtype)
    verts["x"] = points[:, 0]
    verts["y"] = points[:, 1]
    verts["z"] = points[:, 2]
    verts["red"] = colors[:, 0]
    verts["green"] = colors[:, 1]
    verts["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(out_path))

    print(f"[pano_depth] Saved {points.shape[0]} points → {out_path}")

    if save_depth_viz:
        viz = depth - depth.min()
        if viz.max() > 0:
            viz = (viz / viz.max() * 255).astype(np.uint8)
        Image.fromarray(viz, mode="L").save(out_path.with_suffix(".depth.png"))
        print(f"[pano_depth] Saved depth viz → {out_path.with_suffix('.depth.png')}")

    return str(out_path), points, colors


# ─── CLI ─────────────────────────────────────────────────────────────────────

<<<<<<< HEAD
def main():
    parser = argparse.ArgumentParser(description="Panorama → depth → point cloud (for 3DGS init)")
    parser.add_argument("panorama", type=str, help="Path to equirectangular panorama image")
    parser.add_argument("--output", type=str, default=None, help="Output .ply path")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="HuggingFace depth model")
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fallback-if-no-weights", action="store_true",
                        help="Use unit-sphere fallback if model unavailable (keeps plumbing runnable)")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip depth model entirely; use unit-sphere depth. Plumbing check only.")
=======

def main():
    parser = argparse.ArgumentParser(
        description="Panorama → depth → point cloud (for 3DGS init)"
    )
    parser.add_argument(
        "panorama", type=str, help="Path to equirectangular panorama image"
    )
    parser.add_argument("--output", type=str, default=None, help="Output .ply path")
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace depth model",
    )
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fallback-if-no-weights",
        action="store_true",
        help="Use unit-sphere fallback if model unavailable (keeps plumbing runnable)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip depth model entirely; use unit-sphere depth. Plumbing check only.",
    )
>>>>>>> main
    parser.add_argument("--no-depth-viz", action="store_true")
    args = parser.parse_args()

    pano_path = Path(args.panorama)
    if not pano_path.exists():
        print(f"ERROR: panorama not found: {pano_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_ply = Path(args.output)
    else:
        scene = pano_path.stem
        out_ply = pano_path.parent.parent / "pointclouds" / f"{scene}.ply"

    pano_to_pointcloud(
<<<<<<< HEAD
        str(pano_path), str(out_ply),
=======
        str(pano_path),
        str(out_ply),
>>>>>>> main
        depth_model=args.model,
        max_points=args.max_points,
        fallback_if_no_weights=args.fallback_if_no_weights,
        use_model=not args.no_model,
        seed=args.seed,
        save_depth_viz=not args.no_depth_viz,
    )


if __name__ == "__main__":
    main()
