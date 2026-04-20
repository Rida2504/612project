"""
lp3d_layer_gen.py — adapter from our SDXL panorama to LP3D's layered data format.

Orchestrates LP3D's existing scripts (gen_panodepth.py, gen_autolayering.py,
gen_layerdata.py, gen_traindata.py) via subprocess, feeding our SDXL pano as input.

Output layout (matches LP3D's expected structure):
    <out_dir>/
    ├── rgb.png                   # our SDXL pano (copied)
    ├── layering/
    │   ├── depth.npy             # from gen_panodepth (LP3D 360monodepth)
    │   ├── depth_rgb.png
    │   ├── pcd_rgb.ply
    │   ├── rgb.png
    │   └── layerdata.json        # from gen_autolayering + gen_layerdata
    ├── layerdata/                # per-layer RGB + mask + depth
    ├── traindata/                # Flux-inpainted training views
    └── scene/                    # trainer output (populated by P7)

Requires on PATH / env:
    - LP3D repo at $LP3D_ROOT (default $HOME/scratch/phase4/textworld-vr/shared/LayerPano3D)
    - LP3D python deps installed (see P2)
    - Model checkpoints pre-staged (P5, prestage_lp3d_aux.sh)
    - CUDA-capable GPU (H100)
"""

from __future__ import annotations

import argparse
import json
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
LP3D_PYTHON = os.environ.get("LP3D_PYTHON", sys.executable)


def _run_lp3d_stage(
    stage_name: str,
    args: list[str],
    cwd: str = LP3D_ROOT,
    env: dict | None = None,
    timeout_s: int = 1800,
) -> int:
    """Run a LP3D stage script; return exit code."""
    t0 = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    # Force torch/cuda lib path resolution
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        ld = full_env.get("LD_LIBRARY_PATH", "")
        full_env["LD_LIBRARY_PATH"] = f"{torch_lib}:{ld}" if ld else torch_lib
    except ImportError:
        pass

    print(f"\n[lp3d_layer_gen] STAGE {stage_name}: {' '.join(args)}", flush=True)
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=full_env,
        timeout=timeout_s,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    dt = time.time() - t0
    print(
        f"[lp3d_layer_gen] STAGE {stage_name} exit={proc.returncode} t={dt:.1f}s",
        flush=True,
    )
    return proc.returncode


def run_layering(
    pano_path: str,
    out_dir: str,
    n_layers: int = 4,
    scene_type: str = "indoor",
    skip_flux: bool = False,
) -> str:
    """
    Run the LP3D layering pipeline on a single panorama.

    Args:
        pano_path: path to our SDXL panorama .png (2048x1024 equirect).
        out_dir: directory to write layering outputs.
        n_layers: how many depth layers to cluster into.
        scene_type: "indoor" or "outdoor" (affects LP3D's layering heuristics).
        skip_flux: if True, skip gen_traindata (FLUX inpainting). Faster smoke, but
                   trainer will have no back-layer content to supervise.

    Returns: absolute path to the LP3D-format layering directory (suitable for
             passing to run_layerpano.py as --input_dir).
    """
    out_dir = str(Path(out_dir).absolute())
    os.makedirs(out_dir, exist_ok=True)

    # Stage 1: copy our panorama in so LP3D scripts can find it.
    rgb_dst = Path(out_dir) / "rgb.png"
    shutil.copy(pano_path, rgb_dst)
    print(f"[lp3d_layer_gen] copied {pano_path} -> {rgb_dst}")

    # Stage 2: pano-depth via LP3D's own gen_panodepth (360monodepth DA-v2)
    rc = _run_lp3d_stage(
        "panodepth",
        [
            LP3D_PYTHON,
            "gen_panodepth.py",
            "--input_path",
            str(rgb_dst),
            "--save_dir",
            out_dir,
        ],
    )
    if rc != 0:
        raise RuntimeError(f"gen_panodepth failed rc={rc}")
    layering_dir = Path(out_dir) / "layering"
    if not (layering_dir / "depth.npy").exists():
        raise RuntimeError(f"gen_panodepth did not produce depth.npy in {layering_dir}")

    # Stage 3: auto-layering (panoptic seg + KMeans over depth)
    rc = _run_lp3d_stage(
        "autolayering",
        [
            LP3D_PYTHON,
            "gen_autolayering.py",
            "--input_dir",
            out_dir,
            "--scene_type",
            scene_type,
        ],
        timeout_s=900,
    )
    if rc != 0:
        print(
            f"[lp3d_layer_gen] WARNING: gen_autolayering rc={rc}; continuing with depth-only layers",
            flush=True,
        )
        # Minimal fallback: write a trivial layerdata.json with n_layers depth quantiles
        _write_fallback_layerdata(layering_dir, n_layers)

    # Stage 4: build layer data (SAM+LaMa+LLaVA+FLUX-Fill on layer{0,1,2} dirs)
    # LP3D's gen_layerdata takes --base_dir pointing at the layering/ subdir
    # (which contains layer0/, layer1/, layer2/ from gen_autolayering).
    rc = _run_lp3d_stage(
        "layerdata",
        [
            LP3D_PYTHON,
            "gen_layerdata.py",
            "--base_dir",
            str(layering_dir),
        ],
        timeout_s=3600,
    )
    if rc != 0:
        print(f"[lp3d_layer_gen] WARNING: gen_layerdata rc={rc}", flush=True)

    # Stage 5 (optional): train-data (Infusion depth-conditioned inpainting)
    # gen_traindata uses --layerpano_dir (points at layering/) and --save_dir
    # (points at where traindata/ will land, typically out_dir).
    if not skip_flux:
        rc = _run_lp3d_stage(
            "traindata",
            [
                LP3D_PYTHON,
                "gen_traindata.py",
                "--layerpano_dir",
                str(layering_dir),
                "--save_dir",
                out_dir,
                "--root",
                out_dir,
            ],
            timeout_s=3600,
        )
        if rc != 0:
            print(
                f"[lp3d_layer_gen] WARNING: gen_traindata rc={rc} — proceeding without Infusion-inpainted back layers",
                flush=True,
            )

    return str(layering_dir)


def _write_fallback_layerdata(layering_dir: Path, n_layers: int):
    """Trivial fallback: KMeans-less depth quantile layering if gen_autolayering fails."""
    import numpy as np

    depth = np.load(layering_dir / "depth.npy")
    d_min, d_max = float(depth.min()), float(depth.max())
    edges = [d_min + (d_max - d_min) * i / n_layers for i in range(n_layers + 1)]
    layers = []
    for i in range(n_layers):
        mask = ((depth >= edges[i]) & (depth < edges[i + 1])).astype("uint8") * 255
        mask_path = layering_dir / f"mask_layer_{i}.png"
        from PIL import Image

        Image.fromarray(mask).save(mask_path)
        layers.append(
            {
                "id": i,
                "depth_min": edges[i],
                "depth_max": edges[i + 1],
                "mask": str(mask_path),
            }
        )
    (layering_dir / "layerdata.json").write_text(json.dumps(layers, indent=2))
    print(
        f"[lp3d_layer_gen] fallback layerdata.json written with {n_layers} depth-quantile layers",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True, help="path to SDXL panorama .png")
    ap.add_argument("--out-dir", required=True, help="output dir for LP3D layered data")
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--scene-type", default="indoor", choices=["indoor", "outdoor"])
    ap.add_argument(
        "--skip-flux",
        action="store_true",
        help="skip gen_traindata FLUX inpainting (faster smoke)",
    )
    args = ap.parse_args()
    d = run_layering(
        args.pano, args.out_dir, args.n_layers, args.scene_type, args.skip_flux
    )
    print(f"LP3D_LAYER_DIR={d}")


if __name__ == "__main__":
    main()
