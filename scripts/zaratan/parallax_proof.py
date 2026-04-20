"""
parallax_proof.py — render 2 views (origin + offset) from a .ply using gsplat.
Emits origin.png, offset.png, and diff.png (pixel difference heatmap).

Usage:
  python parallax_proof.py --ply <path> --out-dir <dir> \
      [--offset 0.3] [--yaw 0] [--width 1024] [--height 512]

For the LAYERED vs V2 comparison, run twice (once per .ply) and diff the
offset.png frames to see parallax differences between pipelines.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_ply_to_gs(ply_path: str, device: str = "cuda"):
    """Load an INRIA-format .ply into gsplat tensors."""
    from plyfile import PlyData
    pd = PlyData.read(ply_path)
    v = pd["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    # Shs: f_dc_0..2 (DC), plus f_rest_*
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rest_keys = sorted(
        [k for k in v.dtype.names if k.startswith("f_rest_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    rest = np.stack([v[k] for k in rest_keys], axis=1).astype(np.float32) if rest_keys else None
    opacity = v["opacity"].astype(np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    means = torch.from_numpy(xyz).to(device)
    scales = torch.from_numpy(scale).to(device).exp()  # log-space → linear
    quats = torch.from_numpy(rot).to(device)
    opacities = torch.sigmoid(torch.from_numpy(opacity).to(device))
    # SH: pack as (N, K, 3); K = 1 + (len(rest) / 3)
    dc_t = torch.from_numpy(dc).to(device).unsqueeze(1)  # (N, 1, 3)
    if rest is not None and rest.size > 0:
        rest_t = torch.from_numpy(rest.reshape(-1, rest.shape[1] // 3, 3)).to(device)
        sh = torch.cat([dc_t, rest_t], dim=1)  # (N, K, 3)
    else:
        sh = dc_t
    return dict(means=means, scales=scales, quats=quats, opacities=opacities, sh=sh)


def camera_at(position, look_at=(0, 0, 0), up=(0, 1, 0), device: str = "cuda"):
    """Build a view matrix (world-to-camera) at position looking at target."""
    p = np.asarray(position, dtype=np.float32)
    t = np.asarray(look_at, dtype=np.float32)
    u = np.asarray(up, dtype=np.float32)
    f = t - p
    f /= np.linalg.norm(f) + 1e-8
    s = np.cross(f, u)
    s /= np.linalg.norm(s) + 1e-8
    u2 = np.cross(s, f)
    R = np.stack([s, u2, -f], axis=0)  # 3x3
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = -R @ p
    Rt[3, 3] = 1.0
    return torch.from_numpy(Rt).to(device).unsqueeze(0)  # (1, 4, 4)


def render_from(ply_path: str, camera_pos, width: int, height: int, fov_deg: float = 90.0,
                device: str = "cuda", look_at=(0, 0, 10)) -> np.ndarray:
    from gsplat.rendering import rasterization
    gs = load_ply_to_gs(ply_path, device=device)
    view = camera_at(camera_pos, look_at=look_at, device=device)  # (1, 4, 4) world-to-cam
    fx = width / (2 * math.tan(math.radians(fov_deg) / 2))
    fy = height / (2 * math.tan(math.radians(fov_deg) / 2))
    K = torch.tensor(
        [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)
    sh_degree = int(math.isqrt(gs["sh"].shape[1])) - 1
    renders, _, _ = rasterization(
        means=gs["means"], quats=gs["quats"], scales=gs["scales"],
        opacities=gs["opacities"], colors=gs["sh"],
        viewmats=view, Ks=K, width=width, height=height,
        sh_degree=max(0, sh_degree),
    )
    img = renders[0].clamp(0, 1).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--offset", type=float, default=0.3, help="camera translate in x (meters)")
    ap.add_argument("--yaw", type=float, default=0.0, help="camera yaw rotation (degrees)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()

    from PIL import Image
    os.makedirs(args.out_dir, exist_ok=True)

    # Camera is at origin looking at +z; offset view shifts by +x so parallax
    # manifests as lateral motion of near-field content. Look-at at (0,0,10)
    # avoids the degenerate "look at self" view matrix.
    print(f"[proof] rendering origin view (pos=(0,0,0) look=(0,0,10))")
    origin_img = render_from(args.ply, (0, 0, 0), args.width, args.height,
                             look_at=(0, 0, 10))
    Image.fromarray(origin_img).save(os.path.join(args.out_dir, "origin.png"))
    print(f"[proof] rendering offset view (pos=({args.offset},0,0))")
    offset_img = render_from(args.ply, (args.offset, 0, 0), args.width, args.height,
                             look_at=(args.offset, 0, 10))
    Image.fromarray(offset_img).save(os.path.join(args.out_dir, "offset.png"))
    # Pixel-space absolute diff as a heatmap.
    diff = np.abs(origin_img.astype(np.int16) - offset_img.astype(np.int16)).astype(np.uint8)
    Image.fromarray(diff).save(os.path.join(args.out_dir, "diff.png"))
    mean_diff = diff.mean()
    print(f"[proof] mean_pixel_diff={mean_diff:.2f}/255 (higher = more parallax content)")
    print(f"[proof] wrote {args.out_dir}/(origin,offset,diff).png")


if __name__ == "__main__":
    main()
