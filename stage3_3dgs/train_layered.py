"""
train_layered.py — thin wrapper around LP3D's run_layerpano.py trainer.

Consumes the layered data directory produced by stage2_multiview.lp3d_layer_gen
and produces a merged .ply.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


LP3D_ROOT = os.environ.get(
    "LP3D_ROOT",
    "/home/yog/scratch/phase4/textworld-vr/shared/LayerPano3D",
)


def _torch_lib_env() -> dict:
    """Return env dict with LD_LIBRARY_PATH including torch's lib dir."""
    env = os.environ.copy()
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{ld}" if ld else torch_lib
    except ImportError:
        pass
    return env


def run_trainer(
    layered_data_dir: str,
    out_ply: str,
    scene_out_dir: str = None,
    outlier_thresh: int = 4,
    timeout_s: int = 7200,
) -> str:
    """
    Run LP3D's layered 3DGS trainer on a pre-built layered data directory.

    Returns absolute path to the merged .ply output.
    """
    layered_data_dir = str(Path(layered_data_dir).absolute())
    out_ply = str(Path(out_ply).absolute())
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)

    if scene_out_dir is None:
        scene_out_dir = str(Path(out_ply).parent / "lp3d_scene")
    os.makedirs(scene_out_dir, exist_ok=True)

    env = _torch_lib_env()
    t0 = time.time()
    print(
        f"[train_layered] invoking run_layerpano.py input={layered_data_dir} save={scene_out_dir}",
        flush=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "run_layerpano.py",
            "--input_dir",
            layered_data_dir,
            "--save_dir",
            scene_out_dir,
            "--outlier_thresh",
            str(outlier_thresh),
        ],
        cwd=LP3D_ROOT,
        env=env,
        timeout=timeout_s,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    dt = time.time() - t0
    print(
        f"[train_layered] run_layerpano.py exit={proc.returncode} t={dt:.1f}s",
        flush=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"run_layerpano.py failed rc={proc.returncode}")

    # LP3D writes multiple .ply fragments under scene_out_dir/scene. Merge them.
    scene_dir = Path(scene_out_dir) / "scene"
    plys = sorted(scene_dir.rglob("*.ply"))
    if not plys:
        # Some LP3D versions put the final ply elsewhere
        plys = sorted(Path(scene_out_dir).rglob("*.ply"))
    if not plys:
        raise RuntimeError(f"No .ply produced under {scene_out_dir}")
    print(
        f"[train_layered] found {len(plys)} .ply fragments; merging to {out_ply}",
        flush=True,
    )
    _merge_plys(plys, out_ply)
    sz = os.path.getsize(out_ply)
    print(f"[train_layered] merged ply size={sz} ({sz / 1e6:.1f} MB)", flush=True)
    return out_ply


def _merge_plys(input_plys, out_ply):
    """Concatenate multiple LP3D .ply fragments into a single INRIA-format .ply."""
    from plyfile import PlyData, PlyElement
    import numpy as np

    if len(input_plys) == 1:
        shutil.copy(str(input_plys[0]), out_ply)
        return
    vertices = []
    dtype = None
    for p in input_plys:
        pd = PlyData.read(str(p))
        v = pd["vertex"].data
        if dtype is None:
            dtype = v.dtype
        vertices.append(v)
    merged = np.concatenate(vertices)
    # Add a layer_id property if present fragment names suggest layers
    # (heuristic: LP3D names fragments "<N>.ply" or "layer_<N>.ply")
    el = PlyElement.describe(merged, "vertex")
    PlyData([el], text=False).write(out_ply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layerdata", required=True, help="dir produced by lp3d_layer_gen.run_layering"
    )
    ap.add_argument("--out", required=True, help="output .ply path")
    ap.add_argument(
        "--scene-dir",
        default=None,
        help="intermediate scene dir (default: <out_ply_dir>/lp3d_scene)",
    )
    ap.add_argument("--outlier-thresh", type=int, default=4)
    ap.add_argument(
        "--num-iters",
        type=int,
        default=7000,
        help="(informational; passed through env if LP3D respects it)",
    )
    ap.add_argument("--timeout-s", type=int, default=7200)
    args = ap.parse_args()
    # LP3D's trainer reads num_iters from its own arguments.GSParams defaults,
    # not CLI flag. We pass it as LP3D_NUM_ITERS env for future use.
    os.environ["LP3D_NUM_ITERS"] = str(args.num_iters)
    p = run_trainer(
        args.layerdata, args.out, args.scene_dir, args.outlier_thresh, args.timeout_s
    )
    print(f"LAYERED_PLY={p}")


if __name__ == "__main__":
    main()
