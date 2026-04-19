"""Post-process an INRIA-format 3DGS .ply to remove radial thorn artifacts.

Thorns are Gaussians with one scale axis much larger than the others,
stretched along camera view rays (an unavoidable consequence of training
from a single-origin panorama with no parallax).

This script loads the .ply, deletes Gaussians that are (a) too large,
(b) too anisotropic, or (c) too transparent, and rewrites the file.

Usage:
    python -m stage3_3dgs.clean_thorns input.ply output.ply \
        --max-scale 0.03 --max-ratio 1.5 --min-opacity 0.1
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def clean_ply(
    input_path: str,
    output_path: str,
    max_scale: float = 0.03,
    max_ratio: float = 1.5,
    min_opacity: float = 0.1,
    make_isotropic: bool = True,
) -> None:
    ply = PlyData.read(input_path)
    v = ply["vertex"].data
    n0 = len(v)

    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    actual_scale = np.exp(scale)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"]))

    smax = actual_scale.max(axis=1)
    smin = actual_scale.min(axis=1)
    ratio = smax / np.clip(smin, 1e-8, None)

    keep = (smax <= max_scale) & (ratio <= max_ratio) & (opacity >= min_opacity)
    n1 = int(keep.sum())
    print(f"[clean] {n0} -> {n1} Gaussians ({100*(n0-n1)/n0:.1f}% removed)")
    print(f"[clean]   scale>{max_scale}: {int((smax > max_scale).sum())}")
    print(f"[clean]   ratio>{max_ratio}: {int((ratio > max_ratio).sum())}")
    print(f"[clean]   opacity<{min_opacity}: {int((opacity < min_opacity).sum())}")

    v = v[keep]

    if make_isotropic:
        s = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
        s_iso = s.min(axis=1, keepdims=True)
        v["scale_0"] = s_iso[:, 0]
        v["scale_1"] = s_iso[:, 0]
        v["scale_2"] = s_iso[:, 0]

    el = PlyElement.describe(v, "vertex")
    PlyData([el], text=False).write(output_path)
    print(f"[clean] wrote {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_ply")
    ap.add_argument("output_ply")
    ap.add_argument("--max-scale", type=float, default=0.03)
    ap.add_argument("--max-ratio", type=float, default=1.5)
    ap.add_argument("--min-opacity", type=float, default=0.1)
    ap.add_argument("--keep-anisotropy", action="store_true",
                    help="Skip making kept Gaussians isotropic.")
    args = ap.parse_args()
    clean_ply(
        args.input_ply,
        args.output_ply,
        max_scale=args.max_scale,
        max_ratio=args.max_ratio,
        min_opacity=args.min_opacity,
        make_isotropic=not args.keep_anisotropy,
    )


if __name__ == "__main__":
    main()
