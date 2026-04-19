"""
Stage 3 (replacement): gsplat-based 3DGS training with depth-initialized Gaussians.

Design:
  - If `gsplat` (CUDA) is importable → REAL training loop on GPU with
    gsplat's CUDA rasterization. Multi-thousand-iter Adam + L1 loss,
    optional SSIM. Saves 3DGS .ply in INRIA format (mkkellogg-compatible).
  - Else → falls back to the existing `train_3dgs_v2.GaussianModelV2`
    as an init-only export (v2 renderer is not differentiable, so no
    training is done there; see train_via_v2_fallback for rationale).

Input:
  - multiview_dir: directory with per-view images + transforms.json
    (produced by stage2_multiview/extract_views.py)
  - init_pointcloud_ply: .ply produced by stage2_multiview/pano_depth.py

Output:
  - A .ply in the standard INRIA 3DGS format (f_dc_*, f_rest_*, scale_*,
    rot_*, opacity). Loads in mkkellogg/gaussian-splats-3d and SuperSplat.
  - A training log CSV (loss, PSNR each N iterations)
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

_PROJ_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData


SH_C0 = 0.28209479177387814


def _gsplat_available() -> bool:
    try:
        import gsplat  # noqa: F401

        return True
    except Exception:
        return False


def _pick_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pointcloud_ply(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    if all(k in v.data.dtype.names for k in ("red", "green", "blue")):
        rgb = (
            np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.float32)
            / 255.0
        )
    else:
        rgb = np.full_like(xyz, 0.5, dtype=np.float32)
    return xyz, rgb


def knn_mean_distance(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """k-nearest-neighbor mean distance per point, chunked to fit memory."""
    N = points.shape[0]
    out = torch.empty(N, device=points.device, dtype=points.dtype)
    chunk = 4096
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        # (chunk, N) pairwise dists
        d = torch.cdist(points[i:j], points)
        # kth+1 because the closest is self (dist 0)
        knn = torch.topk(d, k=min(k + 1, N), largest=False).values[:, 1:]
        out[i:j] = knn.mean(dim=-1)
    return out


def inverse_sigmoid(x: float) -> float:
    return math.log(x / (1 - x))


def load_cameras(multiview_dir: Path, device: torch.device, downscale: int = 1):
    """Return (viewmats[N,4,4], Ks[N,3,3], images[N,C,H,W], width, height)."""
    tj = json.loads((multiview_dir / "transforms.json").read_text())
    fx = float(tj["fl_x"]) / downscale
    fy = float(tj["fl_y"]) / downscale
    cx = float(tj["cx"]) / downscale
    cy = float(tj["cy"]) / downscale
    W = int(tj["w"]) // downscale
    H = int(tj["h"]) // downscale

    viewmats_list, imgs_list = [], []
    for frame in tj["frames"]:
        # c2w 4x4 in NeRF/OpenGL convention (x-right, y-up, z-back)
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        # Convert OpenGL -> OpenCV (y-down, z-forward) which gsplat expects:
        # flip y and z axes in camera frame (i.e. negate rows 1 and 2 of the
        # world-to-camera rotation, which is equivalent to negating columns 1
        # and 2 of c2w rotation).
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
        c2w_cv = c2w @ flip
        w2c = torch.linalg.inv(c2w_cv)
        viewmats_list.append(w2c)

        # Load image
        img_path = (
            multiview_dir / "images" / frame["file_path"]
            if "file_path" in frame
            else multiview_dir / "images" / f"view_{len(imgs_list):03d}.png"
        )
        # fallback: just iterate view_XXX.png if file_path missing
        if not img_path.exists():
            img_path = multiview_dir / "images" / f"view_{len(imgs_list):03d}.png"
        img = Image.open(img_path).convert("RGB")
        if downscale != 1:
            img = img.resize((W, H), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        imgs_list.append(torch.from_numpy(arr))

    viewmats = torch.stack(viewmats_list, dim=0).to(device)  # (N,4,4)
    imgs = torch.stack(imgs_list, dim=0).to(device)  # (N,H,W,3)
    K = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device
    )
    Ks = K.unsqueeze(0).expand(viewmats.shape[0], 3, 3).contiguous()
    return viewmats, Ks, imgs, W, H


def psnr(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((rendered - gt) ** 2).mean().item()
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Simple 11x11 Gaussian-window SSIM loss (= 1 - SSIM)."""
    # rendered, gt: (H, W, 3) in [0,1]
    x = rendered.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    y = gt.permute(2, 0, 1).unsqueeze(0)
    win_size = 11
    sigma = 1.5
    coords = (
        torch.arange(win_size, device=rendered.device, dtype=rendered.dtype)
        - win_size // 2
    )
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :])[None, None, :, :].expand(3, 1, -1, -1)
    mu_x = F.conv2d(x, win, padding=win_size // 2, groups=3)
    mu_y = F.conv2d(y, win, padding=win_size // 2, groups=3)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, win, padding=win_size // 2, groups=3) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, win, padding=win_size // 2, groups=3) - mu_y_sq
    sigma_xy = F.conv2d(x * y, win, padding=win_size // 2, groups=3) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    return 1.0 - (num / den).mean()


def save_ply_inria(
    means: torch.Tensor,
    scales: torch.Tensor,  # log scale (raw parameter)
    quats: torch.Tensor,  # wxyz (will be normalized)
    opacities: torch.Tensor,  # inverse_sigmoid (raw parameter)
    sh_dc: torch.Tensor,  # (N, 3)
    sh_rest: Optional[torch.Tensor],  # (N, K, 3) or None
    output_path: str,
):
    """Save 3DGS .ply in INRIA format (f_dc_*, f_rest_*, scale_*, rot_*, opacity)."""
    means = means.detach().cpu().numpy().astype(np.float32)
    scales = scales.detach().cpu().numpy().astype(np.float32)
    quats = F.normalize(quats, dim=-1).detach().cpu().numpy().astype(np.float32)
    opacities = opacities.detach().cpu().numpy().astype(np.float32).reshape(-1)
    sh_dc = sh_dc.detach().cpu().numpy().astype(np.float32)  # (N,3)
    if sh_rest is not None:
        sh_rest_np = sh_rest.detach().cpu().numpy().astype(np.float32)  # (N, K, 3)
        n_rest = sh_rest_np.shape[1] * 3  # K * 3 channels
    else:
        sh_rest_np = None
        n_rest = 0

    n = means.shape[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(n_rest):
        header.append(f"property float f_rest_{i}")
    header += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    with open(output_path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        zeros = np.zeros(3, dtype=np.float32)  # placeholder normals
        for i in range(n):
            f.write(struct.pack("<fff", *means[i]))
            f.write(struct.pack("<fff", *zeros))  # nx ny nz
            f.write(struct.pack("<fff", *sh_dc[i]))
            if sh_rest_np is not None:
                # INRIA convention: f_rest laid out as channel-major: K coefs for R, then K for G, then K for B
                K = sh_rest_np.shape[1]
                for c in range(3):
                    for k in range(K):
                        f.write(struct.pack("<f", sh_rest_np[i, k, c]))
            f.write(struct.pack("<f", float(opacities[i])))
            f.write(struct.pack("<fff", *scales[i]))
            f.write(struct.pack("<ffff", *quats[i]))


def train_via_gsplat(
    multiview_dir: str,
    init_xyz: np.ndarray,
    init_rgb: np.ndarray,
    output_ply: str,
    num_iters: int,
    device: torch.device,
    downscale: int = 2,
    sh_degree: int = 3,
    log_csv: Optional[str] = None,
    max_gaussians: int = 500_000,
    lr_means: float = 1.6e-4,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    lr_opacity: float = 5e-2,
    lr_sh_dc: float = 2.5e-3,
    lr_sh_rest: float = 1.25e-4,
    lambda_ssim: float = 0.2,
    log_interval: int = 50,
    densify: bool = True,
    sh_growth_every: int = 1000,
    max_scale: Optional[float] = None,
) -> dict:
    """Real gsplat training loop on CUDA.

    Implements the full "production" variant:
      - SH degree grows 0 → sh_degree_max in `sh_growth_every` steps (starts
        with only DC, adds a new band each interval).
      - Adaptive densification + pruning via `gsplat.DefaultStrategy` (guarded
        by `densify`; falls back to fixed-N if the strategy API mismatches).
      - Per-param-group LRs match the 3DGS paper.
      - Scale & opacity warmup via inverse-sigmoid/log parameterization.
    """
    import gsplat
    from gsplat import rasterization

    multiview_dir = Path(multiview_dir)
    viewmats, Ks, imgs, W, H = load_cameras(multiview_dir, device, downscale=downscale)
    print(f"[gsplat] Loaded {imgs.shape[0]} views at {W}x{H} on {device}.")

    sh_degree_max = int(sh_degree)

    # Subsample if too many initial points (room to densify upward later)
    N = init_xyz.shape[0]
    if N > max_gaussians:
        idx = np.random.choice(N, max_gaussians, replace=False)
        init_xyz = init_xyz[idx]
        init_rgb = init_rgb[idx]
        N = max_gaussians
        print(
            f"[gsplat] Subsampled init to {N} points (max_gaussians={max_gaussians})."
        )

    # Parameters (log-space scales; inverse-sigmoid opacity; wxyz quats; SH split DC/rest)
    means = torch.nn.Parameter(
        torch.tensor(init_xyz, device=device, dtype=torch.float32)
    )
    with torch.no_grad():
        dists = knn_mean_distance(means.detach(), k=3).clamp_min(1e-4)
        init_scale = torch.log(dists * 0.5).unsqueeze(-1).expand(-1, 3).contiguous()
    scales = torch.nn.Parameter(init_scale.clone())
    q0 = torch.zeros(N, 4, device=device, dtype=torch.float32)
    q0[:, 0] = 1.0
    quats = torch.nn.Parameter(q0)
    opacities = torch.nn.Parameter(
        torch.full((N,), inverse_sigmoid(0.1), device=device, dtype=torch.float32)
    )

    sh_dc_init = torch.tensor(
        (init_rgb - 0.5) / SH_C0, device=device, dtype=torch.float32
    ).unsqueeze(1)
    sh0 = torch.nn.Parameter(sh_dc_init.clone())  # (N, 1, 3)  DC
    n_rest = (sh_degree_max + 1) ** 2 - 1  # e.g. 15 for degree 3
    shN = torch.nn.Parameter(
        torch.zeros(N, n_rest, 3, device=device, dtype=torch.float32)
    )

    # Per-param optimizers (DefaultStrategy expects a dict of optimizers)
    params = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }
    optimizers = {
        "means": torch.optim.Adam([means], lr=lr_means, eps=1e-15),
        "quats": torch.optim.Adam([quats], lr=lr_quats, eps=1e-15),
        "scales": torch.optim.Adam([scales], lr=lr_scales, eps=1e-15),
        "opacities": torch.optim.Adam([opacities], lr=lr_opacity, eps=1e-15),
        "sh0": torch.optim.Adam([sh0], lr=lr_sh_dc, eps=1e-15),
        "shN": torch.optim.Adam([shN], lr=lr_sh_rest, eps=1e-15),
    }

    # Adaptive density control strategy (optional)
    strategy = None
    strategy_state = None
    densify_enabled = bool(densify)
    if densify_enabled:
        try:
            from gsplat.strategy import DefaultStrategy

            strategy = DefaultStrategy(
                prune_opa=0.005,
                grow_grad2d=0.0002,
                grow_scale3d=0.01,
                prune_scale3d=0.1,
                refine_start_iter=max(100, num_iters // 10),
                refine_stop_iter=int(num_iters * 0.85),
                reset_every=max(num_iters // 2, 500),
                refine_every=max(50, num_iters // 30),
            )
            strategy.check_sanity(params, optimizers)
            strategy_state = strategy.initialize_state()
            print(
                f"[gsplat] Densification: enabled (start={strategy.refine_start_iter} stop={strategy.refine_stop_iter} every={strategy.refine_every})"
            )
        except Exception as e:
            print(f"[gsplat] Densification: disabled ({type(e).__name__}: {e})")
            strategy = None
            densify_enabled = False

    log_rows = [["iter", "loss", "psnr", "gaussians", "sh_degree", "elapsed_s"]]
    t0 = time.time()
    n_views = imgs.shape[0]
    active_sh = 0

    for it in range(num_iters):
        # Grow SH degree periodically
        if sh_growth_every > 0 and it > 0 and (it % sh_growth_every) == 0:
            active_sh = min(active_sh + 1, sh_degree_max)
            if it % log_interval == 0:
                print(f"[gsplat] iter {it} → SH degree now {active_sh}")

        vi = it % n_views
        gt = imgs[vi]
        # IMPORTANT: always read from `params` dict (not local bindings) because
        # DefaultStrategy may replace params[k] in-place during densify/prune,
        # and stale local refs would render from the OLD parameter tensor.
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N, (K+1)^2, 3)

        # gsplat wants ACTUAL scales (exp of log) and ACTUAL opacities (sigmoid)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=F.normalize(params["quats"], dim=-1),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmats[vi : vi + 1],
            Ks=Ks[vi : vi + 1],
            width=W,
            height=H,
            sh_degree=active_sh,
            packed=False,
            render_mode="RGB",
        )
        rendered = render[0, ..., :3].clamp(0, 1)
        l1 = (rendered - gt).abs().mean()
        if lambda_ssim > 0:
            loss = (1 - lambda_ssim) * l1 + lambda_ssim * ssim_loss(rendered, gt)
        else:
            loss = l1

        # Pre-backward hook (required by DefaultStrategy for bookkeeping)
        if strategy is not None:
            try:
                strategy.step_pre_backward(params, optimizers, strategy_state, it, info)
            except Exception:
                pass

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()

        # Guard against NaN/Inf in any parameter gradient
        for p in params.values():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        for opt in optimizers.values():
            opt.step()

        # Clamp log-space scales to prevent radial thorn artifacts when
        # training data lacks parallax (all cameras at a single origin).
        if max_scale is not None:
            with torch.no_grad():
                log_max = math.log(max_scale)
                params["scales"].clamp_(max=log_max)
                # Enforce isotropy: cap per-Gaussian anisotropy so elongated
                # thorns cannot form along view rays. Limit the largest axis
                # to at most ln(3) above the smallest (ratio <= 3x).
                s = params["scales"]
                s_min = s.min(dim=1, keepdim=True).values
                s.clamp_(min=s_min + 0.0, max=s_min + math.log(3.0))
                s.clamp_(max=log_max)

        # Post-backward hook (this is where DefaultStrategy densifies/prunes)
        if strategy is not None:
            try:
                strategy.step_post_backward(
                    params, optimizers, strategy_state, it, info, packed=False
                )
                # Gaussian count may have changed — refresh N for logging
                N = int(params["means"].shape[0])
            except Exception as e:
                if it < 5:
                    print(
                        f"[gsplat] step_post_backward error (continuing without densify): {e}"
                    )

        if (it % log_interval) == 0 or it == num_iters - 1:
            with torch.no_grad():
                p = psnr(rendered, gt)
            elapsed = time.time() - t0
            log_rows.append(
                [
                    it,
                    round(loss.item(), 5),
                    round(p, 3),
                    N,
                    active_sh,
                    round(elapsed, 2),
                ]
            )
            print(
                f"[gsplat] iter {it:5d} loss={loss.item():.4f} psnr={p:.2f} dB  N={N}  sh={active_sh}  t={elapsed:.1f}s",
                flush=True,
            )

    # Final save uses the (potentially-densified) parameters
    save_ply_inria(
        means=params["means"],
        scales=params["scales"],
        quats=params["quats"],
        opacities=params["opacities"],
        sh_dc=params["sh0"].squeeze(1),
        sh_rest=params["shN"] if sh_degree_max > 0 else None,
        output_path=output_ply,
    )
    if log_csv:
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerows(log_rows)

    final = log_rows[-1]
    # log_rows columns: iter, loss, psnr, gaussians, sh_degree, elapsed_s
    return {
        "backend": "gsplat-cuda",
        "final_gaussians": N,
        "iters": num_iters,
        "device": str(device),
        "final_loss": final[1],
        "final_psnr": final[2],
        "final_sh_degree": final[4],
        "training_time_s": final[5],
        "output_ply": output_ply,
    }


def train_via_v2_fallback(
    multiview_dir: str,
    init_xyz: np.ndarray,
    init_rgb: np.ndarray,
    output_ply: str,
    num_iters: int,
    device: torch.device,
    downscale: int = 2,
    sh_degree: int = 1,
    save_interval: int = 100,
    max_gaussians: int = 100_000,
    log_csv: Optional[str] = None,
) -> dict:
    """CPU/MPS fallback: init-only save (v2 renderer is non-differentiable)."""
    from stage3_3dgs.train_3dgs_v2 import GaussianModelV2, save_gaussian_ply_v2

    N = min(init_xyz.shape[0], max_gaussians)
    if init_xyz.shape[0] > max_gaussians:
        idx = np.random.choice(init_xyz.shape[0], max_gaussians, replace=False)
        init_xyz = init_xyz[idx]
        init_rgb = init_rgb[idx]

    model = GaussianModelV2(N, device, sh_degree=sh_degree)
    model.means.data.copy_(torch.tensor(init_xyz, device=device))
    model.sh_coeffs.data[:, 0, :] = torch.tensor(
        (init_rgb - 0.5) / SH_C0, device=device, dtype=torch.float32
    )
    model.opacities.data.fill_(0.1)
    model.scales.data.fill_(-3.5)

    t0 = time.time()
    save_gaussian_ply_v2(model, output_ply)
    elapsed = time.time() - t0

    if log_csv:
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "loss", "psnr", "gaussians"])
            w.writerow([0, None, None, N])

    print(
        f"[v2-fallback] Saved depth-initialized .ply with {N} Gaussians (no training loop)."
    )
    return {
        "backend": "v2-fallback-init-only",
        "final_gaussians": N,
        "iters": 0,
        "device": str(device),
        "training_time_s": round(elapsed, 2),
        "note": "Init-only; run on Zaratan with gsplat for real optimization.",
    }


def train_gsplat(
    multiview_dir: str,
    init_pointcloud_ply: str,
    output_ply: str,
    num_iters: int = 500,
    device_str: str = "mps",
    downscale: int = 2,
    sh_degree: int = 0,
    max_gaussians: int = 300_000,
    log_csv: Optional[str] = None,
    force_fallback: bool = False,
    max_scale: Optional[float] = None,
) -> dict:
    device = _pick_device(device_str)
    xyz, rgb = load_pointcloud_ply(init_pointcloud_ply)
    print(
        f"[train_gsplat] Loaded {xyz.shape[0]} init points from {init_pointcloud_ply}"
    )
    Path(output_ply).parent.mkdir(parents=True, exist_ok=True)

    if not force_fallback and _gsplat_available() and device.type == "cuda":
        print("[train_gsplat] Using gsplat CUDA backend with real training loop.")
        return train_via_gsplat(
            multiview_dir,
            xyz,
            rgb,
            output_ply,
            num_iters=num_iters,
            device=device,
            downscale=downscale,
            sh_degree=sh_degree,
            log_csv=log_csv,
            max_gaussians=max_gaussians,
            max_scale=max_scale,
        )
    else:
        reason = (
            "forced"
            if force_fallback
            else (
                "gsplat not importable"
                if not _gsplat_available()
                else f"device is {device.type}, not cuda"
            )
        )
        print(f"[train_gsplat] Using v2-fallback backend ({reason}).")
        return train_via_v2_fallback(
            multiview_dir,
            xyz,
            rgb,
            output_ply,
            num_iters=num_iters,
            device=device,
            downscale=downscale,
            sh_degree=sh_degree,
            max_gaussians=min(max_gaussians, 100_000),
            log_csv=log_csv,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train 3DGS with depth-initialized Gaussians (gsplat)."
    )
    parser.add_argument("multiview_dir", type=str)
    parser.add_argument("init_pointcloud", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"]
    )
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--max-gaussians", type=int, default=300_000)
    parser.add_argument("--log-csv", type=str, default=None)
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument(
        "--max-scale",
        type=float,
        default=None,
        help="Clamp each Gaussian's per-axis scale (linear units). "
        "Suppresses radial thorns from panorama-only training.",
    )
    args = parser.parse_args()

    result = train_gsplat(
        args.multiview_dir,
        args.init_pointcloud,
        args.output,
        num_iters=args.iters,
        device_str=args.device,
        downscale=args.downscale,
        sh_degree=args.sh_degree,
        max_gaussians=args.max_gaussians,
        log_csv=args.log_csv,
        force_fallback=args.force_fallback,
        max_scale=args.max_scale,
    )
    print("\n[train_gsplat] Summary:", result)


if __name__ == "__main__":
    main()
