"""
merge_layered_plys.py — concatenate per-layer .ply fragments into a single
INRIA-format .ply for viewing in mkkellogg/gaussian-splats-3d.

Usage:
  python merge_layered_plys.py --scene-dir <lp_trainer/NAME> --out <NAME.ply>
where <lp_trainer/NAME>/scene/gsplat_layer{0..N}.ply exist.

Emits one .ply with vertices concatenated in back-to-front layer order
(important for correct alpha composition during WebGL rendering).
"""
<<<<<<< HEAD
=======

>>>>>>> main
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def merge(scene_dir: str, out_ply: str) -> None:
    sd = Path(scene_dir)
    # Prefer scene/ subdir, fall back to scene_dir itself.
    search_root = sd / "scene" if (sd / "scene").is_dir() else sd
    fragments = sorted(search_root.rglob("gsplat_layer*.ply"))
    if not fragments:
        print(f"ERROR: no gsplat_layer*.ply under {search_root}", file=sys.stderr)
        sys.exit(1)
    print(f"[merge] {len(fragments)} fragments under {search_root}")
    vertices = []
    dtype = None
    total_pts = 0
    for p in fragments:
        pd = PlyData.read(str(p))
        v = pd["vertex"].data
        total_pts += len(v)
        if dtype is None:
            dtype = v.dtype
        vertices.append(v)
        print(f"[merge] {p.name}: {len(v)} points")
    merged = np.concatenate(vertices)
    os.makedirs(os.path.dirname(os.path.abspath(out_ply)), exist_ok=True)
    el = PlyElement.describe(merged, "vertex")
    PlyData([el], text=False).write(out_ply)
    sz = os.path.getsize(out_ply)
    print(f"[merge] out={out_ply} size={sz/1e6:.1f} MB points={len(merged)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
<<<<<<< HEAD
    ap.add_argument("--scene-dir", required=True, help="dir containing scene/gsplat_layer*.ply")
=======
    ap.add_argument(
        "--scene-dir", required=True, help="dir containing scene/gsplat_layer*.ply"
    )
>>>>>>> main
    ap.add_argument("--out", required=True, help="merged .ply output path")
    args = ap.parse_args()
    merge(args.scene_dir, args.out)
