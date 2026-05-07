"""
Stage 3 v2: Enhanced 3D Gaussian Splatting Trainer (Pure PyTorch)

Improvements over v1:
  - Spherical Harmonics (degree 0-3) for view-dependent color
  - Adaptive densification (clone + split high-gradient Gaussians)
  - Opacity-based pruning (remove transparent Gaussians)
  - SSIM loss for perceptual quality
  - Better initialization from depth estimation
  - Progress visualization (renders saved every N steps)

Compatible with MPS (M4 Mac), CUDA (Zaratan A100/H100), and CPU.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# ─── Spherical Harmonics helpers ─────────────────────────────────────────────

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435]


def eval_sh(deg: int, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics at given directions.
    sh_coeffs: (N, C, 3) where C = (deg+1)^2
    dirs: (N, 3) normalized direction vectors
    Returns: (N, 3) RGB colors
    """
    result = SH_C0 * sh_coeffs[:, 0]

    if deg >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result + SH_C1 * (-y * sh_coeffs[:, 1] + z * sh_coeffs[:, 2] - x * sh_coeffs[:, 3])

    if deg >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (
            SH_C2[0] * xy * sh_coeffs[:, 4] +
            SH_C2[1] * yz * sh_coeffs[:, 5] +
            SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6] +
            SH_C2[3] * xz * sh_coeffs[:, 7] +
            SH_C2[4] * (xx - yy) * sh_coeffs[:, 8]
        )

    if deg >= 3:
        result = result + (
            SH_C3[0] * y * (3 * xx - yy) * sh_coeffs[:, 9] +
            SH_C3[1] * xy * z * sh_coeffs[:, 10] +
            SH_C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[:, 11] +
            SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[:, 12] +
            SH_C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[:, 13] +
            SH_C3[5] * z * (xx - yy) * sh_coeffs[:, 14] +
            SH_C3[6] * x * (xx - 3 * yy) * sh_coeffs[:, 15]
        )

    return result + 0.5  # shift from [-0.5, 0.5] to [0, 1]


# ─── SSIM Loss ───────────────────────────────────────────────────────────────

def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute 1 - SSIM between two images (H, W, C) as a differentiable loss."""
    # Reshape to (1, C, H, W)
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)
    C = x.shape[1]

    # Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return 1.0 - ssim_map.mean()


# ─── Gaussian Model ──────────────────────────────────────────────────────────

class GaussianModelV2(nn.Module):
    """Enhanced 3D Gaussian Splatting model with SH colors and densification."""

    def __init__(self, num_points: int, device: torch.device, sh_degree: int = 3):
        super().__init__()
        self.device = device
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.active_sh_degree = 0  # start with degree 0, increase during training

        # Gaussian parameters
        self.means = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.5)
        self.scales = nn.Parameter(torch.ones(num_points, 3, device=device) * -3.0)
        self.rotations = nn.Parameter(torch.zeros(num_points, 4, device=device))
        self.rotations.data[:, 0] = 1.0
        self.opacities = nn.Parameter(torch.zeros(num_points, 1, device=device))

        # SH coefficients: (N, num_sh_coeffs, 3)
        self.sh_coeffs = nn.Parameter(
            torch.zeros(num_points, self.num_sh_coeffs, 3, device=device)
        )
        # Initialize DC component to gray
        self.sh_coeffs.data[:, 0, :] = 0.0

        # Densification tracking
        self.register_buffer(
            'grad_accum', torch.zeros(num_points, device=device)
        )
        self.register_buffer(
            'grad_count', torch.zeros(num_points, device=device, dtype=torch.long)
        )

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    def get_scales(self):
        return torch.exp(self.scales)

    def get_opacities(self):
        return torch.sigmoid(self.opacities)

    def get_colors(self, viewdirs: torch.Tensor | None = None) -> torch.Tensor:
        """Get colors, optionally view-dependent via SH."""
        if viewdirs is not None and self.active_sh_degree > 0:
            return eval_sh(self.active_sh_degree, self.sh_coeffs, viewdirs).clamp(0, 1)
        else:
            # Just DC term
            return (SH_C0 * self.sh_coeffs[:, 0] + 0.5).clamp(0, 1)

    def get_rotation_matrices(self):
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
        return R

    def accumulate_gradients(self):
        """Track gradient magnitudes for densification decisions."""
        if self.means.grad is not None:
            grad_norm = self.means.grad.norm(dim=-1)
            self.grad_accum += grad_norm
            self.grad_count += 1

    def densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        max_screen_size: float = 0.1,
        max_gaussians: int = 100000,
    ):
        """
        Adaptive density control:
        1. Clone small Gaussians with high gradients (under-reconstruction)
        2. Split large Gaussians with high gradients (over-reconstruction)
        3. Prune nearly transparent Gaussians
        """
        if self.grad_count.sum() == 0:
            return 0, 0

        avg_grad = self.grad_accum / (self.grad_count.clamp(min=1))
        scales = self.get_scales()
        opacities = self.get_opacities().squeeze(-1)

        # Mask: high gradient
        high_grad = avg_grad > grad_threshold

        # Clone: small Gaussians with high gradient
        avg_scale = scales.mean(dim=-1)
        clone_mask = high_grad & (avg_scale < 0.01)

        # Split: large Gaussians with high gradient
        split_mask = high_grad & (avg_scale >= 0.01)

        # Prune: low opacity
        prune_mask = opacities < min_opacity

        n_clone = clone_mask.sum().item()
        n_split = split_mask.sum().item()
        n_prune = prune_mask.sum().item()

        # Limit total Gaussians
        new_total = self.num_gaussians + n_clone + n_split - n_prune
        if new_total > max_gaussians:
            # Skip densification if too many
            n_clone = 0
            n_split = 0
            clone_mask[:] = False
            split_mask[:] = False

        new_params = {}

        # Collect all surviving + new Gaussians
        keep_mask = ~prune_mask
        all_means = [self.means.data[keep_mask]]
        all_scales = [self.scales.data[keep_mask]]
        all_rotations = [self.rotations.data[keep_mask]]
        all_opacities = [self.opacities.data[keep_mask]]
        all_sh = [self.sh_coeffs.data[keep_mask]]

        if n_clone > 0:
            all_means.append(self.means.data[clone_mask])
            all_scales.append(self.scales.data[clone_mask])
            all_rotations.append(self.rotations.data[clone_mask])
            all_opacities.append(self.opacities.data[clone_mask])
            all_sh.append(self.sh_coeffs.data[clone_mask])

        if n_split > 0:
            # Split by adding noise to position
            split_means = self.means.data[split_mask]
            noise = torch.randn_like(split_means) * scales[split_mask].mean(dim=-1, keepdim=True)
            all_means.append(split_means + noise)
            all_scales.append(self.scales.data[split_mask] - math.log(1.6))  # smaller
            all_rotations.append(self.rotations.data[split_mask])
            all_opacities.append(self.opacities.data[split_mask])
            all_sh.append(self.sh_coeffs.data[split_mask])

        # Rebuild parameters
        new_n = sum(m.shape[0] for m in all_means)
        self.means = nn.Parameter(torch.cat(all_means, dim=0))
        self.scales = nn.Parameter(torch.cat(all_scales, dim=0))
        self.rotations = nn.Parameter(torch.cat(all_rotations, dim=0))
        self.opacities = nn.Parameter(torch.cat(all_opacities, dim=0))
        self.sh_coeffs = nn.Parameter(torch.cat(all_sh, dim=0))

        # Reset tracking buffers
        self.grad_accum = torch.zeros(new_n, device=self.device)
        self.grad_count = torch.zeros(new_n, device=self.device, dtype=torch.long)

        return n_clone + n_split, n_prune


# ─── Renderer ─────────────────────────────────────────────────────────────────

def render_gaussians(
    model: GaussianModelV2,
    c2w: torch.Tensor,
    fx: float, fy: float,
    cx: float, cy: float,
    width: int, height: int,
    bg_color: tuple = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Render Gaussians with SH view-dependent color. Returns (H, W, 3)."""
    device = model.device

    w2c = torch.inverse(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    means_world = model.means
    means_cam = (R @ means_world.T).T + t
    depth = means_cam[:, 2]
    valid = depth > 0.01

    if valid.sum() == 0:
        bg = torch.tensor(bg_color, device=device).view(1, 1, 3).expand(height, width, 3)
        return bg

    means_cam_v = means_cam[valid]
    depth_v = depth[valid]
    scales_v = model.get_scales()[valid]
    opacities_v = model.get_opacities()[valid]

    # View directions for SH (world space, from camera to Gaussian)
    cam_pos = c2w[:3, 3]
    viewdirs = F.normalize(means_world[valid] - cam_pos.unsqueeze(0), dim=-1)
    colors_v = model.get_colors(viewdirs)

    # Project to pixels
    px = means_cam_v[:, 0] * fx / depth_v + cx
    py = means_cam_v[:, 1] * fy / depth_v + cy

    avg_scale = scales_v.mean(dim=-1)
    radius_px = (avg_scale * fx / depth_v).clamp(min=1.0, max=200.0)

    # Sort back-to-front
    sort_idx = torch.argsort(depth_v, descending=True)
    px = px[sort_idx]
    py = py[sort_idx]
    colors_v = colors_v[sort_idx]
    opacities_v = opacities_v[sort_idx]
    radius_px = radius_px[sort_idx]

    # Rasterize
    bg_tensor = torch.tensor(bg_color, device=device).view(1, 1, 3)
    image = bg_tensor.expand(height, width, 3).clone()
    alpha_acc = torch.zeros(height, width, 1, device=device)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij',
    )

    max_splats = min(len(px), 5000)  # increased limit
    for i in range(max_splats):
        x0, y0 = px[i], py[i]
        r = radius_px[i]
        alpha = opacities_v[i]
        color = colors_v[i]

        x_min = max(0, int(x0 - r * 3))
        x_max = min(width, int(x0 + r * 3) + 1)
        y_min = max(0, int(y0 - r * 3))
        y_max = min(height, int(y0 + r * 3) + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        dx = xx[y_min:y_max, x_min:x_max] - x0
        dy = yy[y_min:y_max, x_min:x_max] - y0
        gauss = torch.exp(-0.5 * (dx**2 + dy**2) / (r**2 + 1e-6))

        a = (gauss * alpha).unsqueeze(-1)
        remaining = 1.0 - alpha_acc[y_min:y_max, x_min:x_max]
        contribution = a * remaining
        image[y_min:y_max, x_min:x_max] += contribution * (color - bg_tensor.squeeze(0))
        alpha_acc[y_min:y_max, x_min:x_max] += contribution

    return image.clamp(0, 1)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_training_data(multiview_dir: str, device: torch.device, downscale: int = 1):
    """Load images and camera poses from a multi-view directory."""
    mv_path = Path(multiview_dir)
    with open(mv_path / "transforms.json") as f:
        transforms = json.load(f)

    images = []
    cameras = []

    for frame in transforms["frames"]:
        img_path = mv_path / frame["file_path"]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if downscale > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // downscale, h // downscale))

        img_tensor = torch.from_numpy(img).float().to(device) / 255.0
        images.append(img_tensor)

        h, w = img_tensor.shape[:2]
        fl_x = frame.get("fl_x", transforms.get("fl_x", w / 2))
        fl_y = frame.get("fl_y", transforms.get("fl_y", h / 2))
        cx_val = frame.get("cx", transforms.get("cx", w / 2))
        cy_val = frame.get("cy", transforms.get("cy", h / 2))

        if downscale > 1:
            fl_x /= downscale
            fl_y /= downscale
            cx_val /= downscale
            cy_val /= downscale

        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)

        cameras.append({
            "c2w": c2w, "fx": fl_x, "fy": fl_y,
            "cx": cx_val, "cy": cy_val, "w": w, "h": h,
        })

    return images, cameras


def initialize_from_views(cameras: list, num_points: int, device: torch.device) -> torch.Tensor:
    """Initialize Gaussian positions using camera frustum sampling."""
    all_points = []

    for cam in cameras:
        c2w = cam["c2w"]
        cam_pos = c2w[:3, 3].cpu().numpy()

        # Sample points along camera rays at various depths
        for depth in [0.5, 1.0, 2.0, 3.0, 5.0]:
            n = num_points // (len(cameras) * 5) + 1
            # Random pixel coordinates
            us = np.random.uniform(0, cam["w"], n)
            vs = np.random.uniform(0, cam["h"], n)

            # Unproject
            x = (us - cam["cx"]) / cam["fx"] * depth
            y = (vs - cam["cy"]) / cam["fy"] * depth
            z = np.full(n, depth)

            pts_cam = np.stack([x, y, z], axis=-1)

            # Transform to world
            R = c2w[:3, :3].cpu().numpy()
            t = c2w[:3, 3].cpu().numpy()
            pts_world = (R @ pts_cam.T).T + t
            all_points.append(pts_world)

    all_points = np.concatenate(all_points, axis=0)

    # Subsample to desired number
    if len(all_points) > num_points:
        idx = np.random.choice(len(all_points), num_points, replace=False)
        all_points = all_points[idx]
    elif len(all_points) < num_points:
        # Pad with random
        extra = np.random.randn(num_points - len(all_points), 3) * 0.5
        all_points = np.concatenate([all_points, extra], axis=0)

    return torch.tensor(all_points, dtype=torch.float32, device=device)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    multiview_dir: str,
    output_ply: str,
    num_iters: int = 3000,
    num_points: int = 10000,
    lr: float = 0.005,
    downscale: int = 2,
    device_str: str = "mps",
    sh_degree: int = 3,
    densify_interval: int = 200,
    densify_start: int = 500,
    densify_stop: int = 2500,
    prune_interval: int = 300,
    save_interval: int = 500,
    lambda_ssim: float = 0.2,
    max_gaussians: int = 50000,
):
    """Train an enhanced 3DGS model from multi-view images."""
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[3DGS v2] Device: {device}")
    print(f"[3DGS v2] SH degree: {sh_degree}, Max Gaussians: {max_gaussians}")

    images, cameras = load_training_data(multiview_dir, device, downscale)
    print(f"[3DGS v2] Loaded {len(images)} views, {images[0].shape[1]}x{images[0].shape[0]}")

    # Initialize model with frustum-sampled points
    model = GaussianModelV2(num_points, device, sh_degree=sh_degree)
    init_pts = initialize_from_views(cameras, num_points, device)
    model.means.data.copy_(init_pts)
    print(f"[3DGS v2] Initialized {num_points} Gaussians from camera frustums")

    # Optimizer - rebuild function for after densification
    def make_optimizer(model):
        return torch.optim.Adam([
            {"params": [model.means], "lr": lr, "name": "means"},
            {"params": [model.scales], "lr": lr * 0.5, "name": "scales"},
            {"params": [model.rotations], "lr": lr * 0.1, "name": "rotations"},
            {"params": [model.opacities], "lr": lr * 0.5, "name": "opacities"},
            {"params": [model.sh_coeffs], "lr": lr * 0.5, "name": "sh_coeffs"},
        ])

    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    # Output directory for progress renders
    output_dir = Path(output_ply).parent
    progress_dir = output_dir / "progress" / Path(output_ply).stem
    progress_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[3DGS v2] Training for {num_iters} iterations...")
    t0 = time.time()
    num_views = len(images)
    losses = []

    for step in range(num_iters):
        # Increase SH degree over time
        if step == 500:
            model.active_sh_degree = min(1, sh_degree)
            print(f"  [SH] Increased to degree {model.active_sh_degree}")
        elif step == 1000:
            model.active_sh_degree = min(2, sh_degree)
            print(f"  [SH] Increased to degree {model.active_sh_degree}")
        elif step == 1500:
            model.active_sh_degree = min(3, sh_degree)
            print(f"  [SH] Increased to degree {model.active_sh_degree}")

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

        # Combined loss: L1 + SSIM
        l1 = F.l1_loss(rendered, gt_image)
        loss = (1.0 - lambda_ssim) * l1

        if lambda_ssim > 0 and step > 100:
            s_loss = ssim_loss(rendered, gt_image)
            loss = loss + lambda_ssim * s_loss

        optimizer.zero_grad()
        loss.backward()

        # Track gradients for densification
        model.accumulate_gradients()

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Densification
        if densify_start <= step < densify_stop and step % densify_interval == 0:
            n_added, n_pruned = model.densify_and_prune(
                max_gaussians=max_gaussians,
            )
            optimizer = make_optimizer(model)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
            if n_added > 0 or n_pruned > 0:
                print(f"  [Densify] Step {step+1}: +{n_added} -{n_pruned} = {model.num_gaussians} Gaussians")

        # Pruning only
        elif step >= densify_stop and step % prune_interval == 0:
            _, n_pruned = model.densify_and_prune(
                grad_threshold=999.0,  # no densification
                max_gaussians=max_gaussians,
            )
            if n_pruned > 0:
                optimizer = make_optimizer(model)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
                print(f"  [Prune] Step {step+1}: -{n_pruned} = {model.num_gaussians} Gaussians")

        # Logging
        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - t0
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f"  Step {step+1}/{num_iters} | Loss: {avg_loss:.4f} | "
                  f"Gaussians: {model.num_gaussians} | Time: {elapsed:.1f}s")

        # Save progress renders
        if save_interval > 0 and (step + 1) % save_interval == 0:
            with torch.no_grad():
                render_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_image.cpu().numpy() * 255).astype(np.uint8)
                comparison = np.concatenate([gt_np, render_np], axis=1)
                comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(progress_dir / f"step_{step+1:05d}.png"), comparison_bgr)

    elapsed = time.time() - t0
    print(f"\n[3DGS v2] Training complete in {elapsed:.1f}s")
    print(f"[3DGS v2] Final Gaussians: {model.num_gaussians}")

    # Save final PLY
    save_gaussian_ply_v2(model, output_ply)

    # Save training stats
    stats = {
        "num_iters": num_iters,
        "final_loss": losses[-1] if losses else 0,
        "avg_loss_last_100": sum(losses[-100:]) / len(losses[-100:]) if losses else 0,
        "final_gaussians": model.num_gaussians,
        "training_time_s": elapsed,
        "sh_degree": sh_degree,
        "device": str(device),
    }
    stats_path = Path(output_ply).with_suffix(".json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[3DGS v2] Stats: {stats_path}")

    return model


def save_gaussian_ply_v2(model: GaussianModelV2, output_path: str):
    """Save Gaussian model to PLY format (compatible with standard viewers)."""
    means = model.means.detach().cpu().numpy()
    scales = model.get_scales().detach().cpu().numpy()
    rotations = F.normalize(model.rotations, dim=-1).detach().cpu().numpy()
    opacities = model.get_opacities().detach().cpu().numpy()
    # Use DC color for PLY (view-independent)
    colors = model.get_colors(None).detach().cpu().clamp(0, 1).numpy()
    colors_uint8 = (colors * 255).astype(np.uint8)
    n = means.shape[0]

    # Also save SH coefficients for advanced viewers
    sh = model.sh_coeffs.detach().cpu().numpy()  # (N, C, 3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Standard PLY with basic properties (compatible with most viewers)
    with open(output_path, "wb") as f:
        # Build header with SH properties
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "property float opacity",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        # Add SH coefficient properties
        num_sh = model.num_sh_coeffs
        for c in range(3):  # RGB
            for j in range(num_sh):
                header_lines.append(f"property float f_rest_{c * num_sh + j}")
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"
        f.write(header.encode("ascii"))

        for i in range(n):
            # Position
            f.write(struct.pack("<fff", means[i, 0], means[i, 1], means[i, 2]))
            # Scale (log)
            f.write(struct.pack("<fff",
                np.log(scales[i, 0]), np.log(scales[i, 1]), np.log(scales[i, 2])))
            # Rotation (quaternion)
            f.write(struct.pack("<ffff",
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]))
            # Opacity (logit)
            op = float(np.log(opacities[i, 0] / (1.0 - opacities[i, 0] + 1e-8)))
            f.write(struct.pack("<f", op))
            # Color
            f.write(struct.pack("<BBB",
                colors_uint8[i, 0], colors_uint8[i, 1], colors_uint8[i, 2]))
            # SH coefficients
            for c in range(3):
                for j in range(num_sh):
                    f.write(struct.pack("<f", sh[i, j, c]))

    print(f"  Saved {n} Gaussians ({num_sh} SH coeffs) to {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train enhanced 3DGS v2 from multi-view images")
    parser.add_argument("multiview_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-iters", type=int, default=3000)
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--max-gaussians", type=int, default=50000)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--save-interval", type=int, default=500)
    args = parser.parse_args()

    scene_name = Path(args.multiview_dir).name
    output_ply = args.output or f"outputs/splats/{scene_name}_v2.ply"

    train(
        args.multiview_dir,
        output_ply,
        num_iters=args.num_iters,
        num_points=args.num_points,
        lr=args.lr,
        downscale=args.downscale,
        device_str=args.device,
        sh_degree=args.sh_degree,
        max_gaussians=args.max_gaussians,
        lambda_ssim=args.lambda_ssim,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
