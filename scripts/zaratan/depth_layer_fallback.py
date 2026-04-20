"""
depth_layer_fallback.py — emit depth-quantile layer dirs when LP3D's panoptic
seg-driven gen_autolayering finds 0 instances.

Invoked only if <layering_dir>/layer0/layer0_mask.png is missing after autolayering.
Reads <layering_dir>/depth.npy and writes N equal-quantile masks into
layerN/ subdirs matching LP3D's filename conventions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def build(layering_dir: str, n_layers: int = 3) -> None:
    ld = Path(layering_dir)
    depth_path = ld / "depth.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"no depth.npy at {depth_path}")
    depth = np.load(depth_path)
    # Normalize depth to [0,1] for quantile splits.
    d_min, d_max = float(depth.min()), float(depth.max())
    depth_n = (depth - d_min) / max(d_max - d_min, 1e-8)
    # Quantile-based splits: equal pixel-mass per layer (stable vs depth histogram skew)
    edges = np.quantile(depth_n, np.linspace(0, 1, n_layers + 1))
    print(f"[fallback] depth range [{d_min:.4f},{d_max:.4f}] quantiles: {edges.tolist()}")
    for i in range(n_layers):
        lo, hi = edges[i], edges[i + 1]
        mask = ((depth_n >= lo) & (depth_n <= hi)).astype(np.uint8) * 255
        layer_dir = ld / f"layer{i}"
        layer_dir.mkdir(exist_ok=True)
        # LP3D expects mask.png + mask_smooth.png (downstream scripts read both)
        Image.fromarray(mask).save(layer_dir / f"layer{i}_mask.png")
        Image.fromarray(mask).save(layer_dir / f"layer{i}_mask_smooth.png")
        px = int(mask.sum() / 255)
        print(f"[fallback] layer{i}: depth[{lo:.3f},{hi:.3f}] = {px} px")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layering-dir", required=True)
    ap.add_argument("--n-layers", type=int, default=3)
    args = ap.parse_args()
    build(args.layering_dir, args.n_layers)
