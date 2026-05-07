"""
Stage 2: Equirectangular 360° Panorama → Multi-View Perspective Images
Extracts N perspective views with known camera poses for 3DGS reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import yaml


def equirect_to_perspective(
    equirect: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Sample a perspective view from an equirectangular panorama.

    Args:
        equirect: Input equirectangular image (H, W, 3)
        fov_deg: Horizontal field of view in degrees
        yaw_deg: Horizontal rotation (0=front, 90=right, etc.)
        pitch_deg: Vertical rotation (0=horizon, +up, -down)
        out_w, out_h: Output image dimensions
    """
    h, w = equirect.shape[:2]
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Focal length from FOV
    f = out_w / (2 * math.tan(fov / 2))

    # Pixel grid for output image
    u = np.arange(out_w) - out_w / 2
    v = np.arange(out_h) - out_h / 2
    u, v = np.meshgrid(u, v)

    # Direction vectors in camera space (looking along +Z)
    x = u
    y = -v  # flip y (image y goes down, world y goes up)
    z = np.full_like(u, f, dtype=np.float64)

    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm

    # Rotation: pitch (around X), then yaw (around Y)
    # Pitch rotation
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    y2 = y * cos_p - z * sin_p
    z2 = y * sin_p + z * cos_p
    y, z = y2, z2

    # Yaw rotation
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x2 = x * cos_y + z * sin_y
    z2 = -x * sin_y + z * cos_y
    x, z = x2, z2

    # Convert to spherical coordinates
    theta = np.arctan2(x, z)  # longitude [-π, π]
    phi = np.arcsin(np.clip(y, -1, 1))  # latitude [-π/2, π/2]

    # Map to equirectangular pixel coordinates
    eq_x = ((theta / math.pi + 1) / 2 * w).astype(np.float32)
    eq_y = ((0.5 - phi / math.pi) * h).astype(np.float32)

    # Wrap horizontally
    eq_x = eq_x % w

    # Sample with bilinear interpolation
    perspective = cv2.remap(
        equirect,
        eq_x,
        eq_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )

    return perspective


def compute_camera_params(
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    img_w: int,
    img_h: int,
    radius: float = 1.0,
) -> dict:
    """
    Compute camera extrinsics and intrinsics for a given view direction.
    Returns a dict compatible with COLMAP/OpenSplat transforms.json format.

    For panoramic scenes, cameras are at origin looking outward (the scene
    surrounds the viewer). A small jitter is added for 3DGS parallax.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    fov = math.radians(fov_deg)

    # Camera is placed on a sphere looking inward toward the center.
    # The radius controls the baseline between views — larger radius
    # gives more parallax for better 3DGS depth estimation.
    # For indoor panoramas, radius ~0.5 simulates slight head movement.
    cx = radius * math.sin(yaw) * math.cos(pitch)
    cy = radius * math.sin(pitch)
    cz = radius * math.cos(yaw) * math.cos(pitch)

    # Forward direction: looking from camera position toward center
    forward = np.array([-cx, -cy, -cz])
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / norm

    # World up
    up = np.array([0.0, 1.0, 0.0])

    # Right
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # 4x4 camera-to-world transform
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # OpenGL convention: camera looks along -Z
    c2w[:3, 3] = [cx, cy, cz]

    # Focal length
    fl_x = img_w / (2 * math.tan(fov / 2))
    fl_y = fl_x  # square pixels

    return {
        "transform_matrix": c2w.tolist(),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": img_w / 2,
        "cy": img_h / 2,
        "w": img_w,
        "h": img_h,
        "camera_angle_x": fov,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
    }


def extract_multiviews(
    panorama_path: str,
    output_dir: str,
    num_views: int = 8,
    fov_deg: float = 90.0,
    out_w: int = 1024,
    out_h: int = 1024,
    elevation_angles: list[float] | None = None,
):
    """Extract perspective views from a panorama and save with camera poses."""
    if elevation_angles is None:
        elevation_angles = [0.0]

    pano = cv2.imread(panorama_path)
    if pano is None:
        raise FileNotFoundError(f"Cannot read panorama: {panorama_path}")

    pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    out = Path(output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    idx = 0
    yaw_step = 360.0 / num_views

    for pitch in elevation_angles:
        for i in range(num_views):
            yaw = i * yaw_step

            # Extract perspective view
            view = equirect_to_perspective(pano_rgb, fov_deg, yaw, pitch, out_w, out_h)

            # Save image
            fname = f"view_{idx:03d}.png"
            view_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(images_dir / fname), view_bgr)

            # Compute camera parameters
            cam = compute_camera_params(yaw, pitch, fov_deg, out_w, out_h)
            cam["file_path"] = f"images/{fname}"
            frames.append(cam)

            print(f"  View {idx}: yaw={yaw:.0f}°, pitch={pitch:.0f}° → {fname}")
            idx += 1

    # Save transforms.json (nerfstudio format, compatible with OpenSplat)
    transforms = {
        "camera_model": "PINHOLE",
        "camera_angle_x": math.radians(fov_deg),
        "fl_x": frames[0]["fl_x"],
        "fl_y": frames[0]["fl_y"],
        "cx": frames[0]["cx"],
        "cy": frames[0]["cy"],
        "w": out_w,
        "h": out_h,
        "frames": frames,
    }

    transforms_path = out / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"  Saved {idx} views + transforms.json → {out}")
    return transforms


def main():
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Extract multi-view images from 360° panorama")
    parser.add_argument("panorama", type=str, help="Path to equirectangular panorama image")
=======
    parser = argparse.ArgumentParser(
        description="Extract multi-view images from 360° panorama"
    )
    parser.add_argument(
        "panorama", type=str, help="Path to equirectangular panorama image"
    )
>>>>>>> main
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--fov", type=float, default=None, help="FOV in degrees")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mv_cfg = config["multiview"]
    num_views = args.num_views or mv_cfg["num_views"]
    fov = args.fov or mv_cfg["fov_degrees"]
    out_w = mv_cfg["output_width"]
    out_h = mv_cfg["output_height"]
    elevations = mv_cfg.get("elevation_angles", [0.0])

    # Output dir
    if args.output:
        output_dir = args.output
    else:
        pano_name = Path(args.panorama).stem
<<<<<<< HEAD
        output_dir = str(Path(__file__).parent.parent / config.get("output_dir", "outputs") / "multiview" / pano_name)

    print(f"Extracting {num_views} views (FOV={fov}°) from: {args.panorama}")
    extract_multiviews(args.panorama, output_dir, num_views, fov, out_w, out_h, elevations)
=======
        output_dir = str(
            Path(__file__).parent.parent
            / config.get("output_dir", "outputs")
            / "multiview"
            / pano_name
        )

    print(f"Extracting {num_views} views (FOV={fov}°) from: {args.panorama}")
    extract_multiviews(
        args.panorama, output_dir, num_views, fov, out_w, out_h, elevations
    )
>>>>>>> main
    print("Done!")


if __name__ == "__main__":
    main()
