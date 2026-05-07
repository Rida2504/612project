"""
Stage 3: Minimal 3D Gaussian Splatting Trainer (Pure PyTorch)

A self-contained 3DGS implementation that works on MPS/CPU without
external C++ dependencies. Uses differentiable rendering with
alpha-compositing for Gaussian splatting.

For production quality, use OpenSplat or gsplat. This trainer is
sufficient for prototyping and demo scenes.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import yaml


class GaussianModel(nn.Module):
    """3D Gaussian Splatting model with learnable parameters."""

    def __init__(self, num_points: int, device: torch.device):
        super().__init__()
        self.device = device

        # Initialize Gaussian parameters
        self.means = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.5)
        self.scales = nn.Parameter(torch.ones(num_points, 3, device=device) * -3.0)  # log scale
        self.rotations = nn.Parameter(torch.zeros(num_points, 4, device=device))
        self.rotations.data[:, 0] = 1.0  # identity quaternion
        self.opacities = nn.Parameter(torch.zeros(num_points, 1, device=device))  # sigmoid
        self.colors = nn.Parameter(torch.rand(num_points, 3, device=device))  # RGB in [0,1]

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    def get_scales(self):
        return torch.exp(self.scales)

    def get_opacities(self):
        return torch.sigmoid(self.opacities)

    def get_rotation_matrices(self):
        """Convert quaternions to rotation matrices."""
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
        return R


def render_gaussians(
    model: GaussianModel,
    c2w: torch.Tensor,
    fx: float, fy: float,
    cx: float, cy: float,
    width: int, height: int,
) -> torch.Tensor:
    """
    Render Gaussians from a given camera viewpoint using splatting.
    Returns an (H, W, 3) image tensor.
    """
    device = model.device

    # World-to-camera transform
    w2c = torch.inverse(c2w)
    R = w2c[:3, :3]  # (3, 3)
    t = w2c[:3, 3]   # (3,)

    # Transform Gaussian means to camera space
    means_world = model.means  # (N, 3)
    means_cam = (R @ means_world.T).T + t  # (N, 3)

    # Filter Gaussians behind camera
    depth = means_cam[:, 2]
    valid = depth > 0.01
    if valid.sum() == 0:
        return torch.zeros(height, width, 3, device=device)

    means_cam = means_cam[valid]
    depth = depth[valid]
    colors = model.colors[valid]
    opacities = model.get_opacities()[valid]
    scales = model.get_scales()[valid]

    # Project to pixel coordinates
    px = means_cam[:, 0] * fx / depth + cx
    py = means_cam[:, 1] * fy / depth + cy

    # Compute 2D Gaussian radius (approximate)
    # Use average scale projected to image
    avg_scale = scales.mean(dim=-1)  # (N,)
    radius_px = avg_scale * fx / depth  # approximate pixel radius

    # Sort by depth (back to front for alpha compositing)
    sort_idx = torch.argsort(depth, descending=True)
    px = px[sort_idx]
    py = py[sort_idx]
    colors = colors[sort_idx]
    opacities = opacities[sort_idx]
    radius_px = radius_px[sort_idx]

    # Rasterize using a tile-based approach
    image = torch.zeros(height, width, 3, device=device)
    alpha_acc = torch.zeros(height, width, 1, device=device)

    # Create pixel grid
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij',
    )

    # Process each Gaussian (vectorized per-gaussian)
    for i in range(min(len(px), 2000)):  # limit for speed
        x0, y0 = px[i], py[i]
        r = radius_px[i].clamp(min=1.0, max=200.0)
        alpha = opacities[i]
        color = colors[i]

        # Bounding box
        x_min = max(0, int(x0 - r * 3))
        x_max = min(width, int(x0 + r * 3) + 1)
        y_min = max(0, int(y0 - r * 3))
        y_max = min(height, int(y0 + r * 3) + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        # Gaussian weight
        dx = xx[y_min:y_max, x_min:x_max] - x0
        dy = yy[y_min:y_max, x_min:x_max] - y0
        gauss = torch.exp(-0.5 * (dx**2 + dy**2) / (r**2 + 1e-6))

        # Alpha compositing
        a = (gauss * alpha).unsqueeze(-1)
        remaining = 1.0 - alpha_acc[y_min:y_max, x_min:x_max]
        contribution = a * remaining
        image[y_min:y_max, x_min:x_max] += contribution * color
        alpha_acc[y_min:y_max, x_min:x_max] += contribution

    return image.clamp(0, 1)


def load_training_data(multiview_dir: str, device: torch.device, downscale: int = 1):
    """Load images and camera poses from a multi-view directory."""
    mv_path = Path(multiview_dir)
    with open(mv_path / "transforms.json") as f:
        transforms = json.load(f)

    images = []
    cameras = []

    for frame in transforms["frames"]:
        # Load image
        img_path = mv_path / frame["file_path"]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if downscale > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // downscale, h // downscale))

        img_tensor = torch.from_numpy(img).float().to(device) / 255.0
        images.append(img_tensor)

        # Camera parameters
        h, w = img_tensor.shape[:2]
        fl_x = frame.get("fl_x", transforms.get("fl_x", w / 2))
        fl_y = frame.get("fl_y", transforms.get("fl_y", h / 2))
        cx = frame.get("cx", transforms.get("cx", w / 2))
        cy = frame.get("cy", transforms.get("cy", h / 2))

        if downscale > 1:
            fl_x /= downscale
            fl_y /= downscale
            cx /= downscale
            cy /= downscale

        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)

        cameras.append({
            "c2w": c2w,
            "fx": fl_x, "fy": fl_y,
            "cx": cx, "cy": cy,
            "w": w, "h": h,
        })

    return images, cameras


def train(
    multiview_dir: str,
    output_ply: str,
    num_iters: int = 2000,
    num_points: int = 5000,
    lr: float = 0.01,
    downscale: int = 2,
    device_str: str = "mps",
):
    """Train a 3DGS model from multi-view images."""
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading training data from: {multiview_dir}")
    images, cameras = load_training_data(multiview_dir, device, downscale)
    print(f"  Loaded {len(images)} views, resolution: {images[0].shape[1]}x{images[0].shape[0]}")

    # Initialize model
    model = GaussianModel(num_points, device)
    print(f"  Initialized {num_points} Gaussians")

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": [model.means], "lr": lr},
        {"params": [model.scales], "lr": lr * 0.5},
        {"params": [model.rotations], "lr": lr * 0.1},
        {"params": [model.opacities], "lr": lr * 0.5},
        {"params": [model.colors], "lr": lr * 2.0},
    ])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

    print(f"\nTraining for {num_iters} iterations...")
    t0 = time.time()
    num_views = len(images)

    for step in range(num_iters):
        # Random view
        idx = step % num_views
        gt_image = images[idx]
        cam = cameras[idx]

        # Render
        rendered = render_gaussians(
            model, cam["c2w"],
            cam["fx"], cam["fy"],
            cam["cx"], cam["cy"],
            cam["w"], cam["h"],
        )

        # Loss: L1 + SSIM-like
        loss = F.l1_loss(rendered, gt_image)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{num_iters} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Save PLY
    save_gaussian_ply(model, output_ply)
    print(f"Saved: {output_ply}")

    return model


def save_gaussian_ply(model: GaussianModel, output_path: str):
    """Save Gaussian model to PLY format compatible with viewers."""
    means = model.means.detach().cpu().numpy()
    scales = model.get_scales().detach().cpu().numpy()
    rotations = F.normalize(model.rotations, dim=-1).detach().cpu().numpy()
    opacities = model.get_opacities().detach().cpu().numpy()
    colors = (model.colors.detach().cpu().clamp(0, 1) * 255).numpy().astype(np.uint8)
    n = means.shape[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "property float rot_3\n"
            "property float opacity\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff",
                means[i, 0], means[i, 1], means[i, 2]))
            f.write(struct.pack("<fff",
                scales[i, 0], scales[i, 1], scales[i, 2]))
            f.write(struct.pack("<ffff",
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]))
            f.write(struct.pack("<f", opacities[i, 0]))
            f.write(struct.pack("<BBB",
                colors[i, 0], colors[i, 1], colors[i, 2]))

    print(f"  Saved {n} Gaussians to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train 3DGS from multi-view images")
    parser.add_argument("multiview_dir", type=str, help="Directory with images/ and transforms.json")
    parser.add_argument("--output", type=str, default=None, help="Output .ply path")
    parser.add_argument("--num-iters", type=int, default=2000)
    parser.add_argument("--num-points", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    scene_name = Path(args.multiview_dir).name
    output_ply = args.output or f"outputs/splats/{scene_name}.ply"

    train(
        args.multiview_dir,
        output_ply,
        num_iters=args.num_iters,
        num_points=args.num_points,
        lr=args.lr,
        downscale=args.downscale,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
