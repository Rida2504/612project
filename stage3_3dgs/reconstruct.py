"""
Stage 3: Multi-View Images → 3D Gaussian Splatting Reconstruction
Uses OpenSplat (Metal/MPS on Mac, CUDA on Linux) to reconstruct a 3DGS scene.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def find_opensplat() -> Optional[str]:
    """Locate the OpenSplat binary."""
    # Check PATH
    path = shutil.which("opensplat")
    if path:
        return path

    # Check local build
    local_build = Path(__file__).parent / "OpenSplat" / "build" / "opensplat"
    if local_build.exists():
        return str(local_build)

    return None


def prepare_colmap_structure(multiview_dir: str, work_dir: str) -> str:
    """
    Convert transforms.json format to COLMAP-style directory structure
    that OpenSplat can read natively.

    OpenSplat expects either:
      - COLMAP format: images/ + sparse/0/{cameras,images,points3D}.{bin,txt}
      - Or a nerfstudio transforms.json

    We'll create a nerfstudio-compatible structure.
    """
    mv_path = Path(multiview_dir)
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    transforms_src = mv_path / "transforms.json"
    if not transforms_src.exists():
        raise FileNotFoundError(f"No transforms.json in {multiview_dir}")

    # Copy transforms.json to work dir
    transforms_dst = work_path / "transforms.json"
    with open(transforms_src) as f:
        transforms = json.load(f)

    # Copy images and update paths to be relative to work_dir
    images_dst = work_path / "images"
    images_dst.mkdir(exist_ok=True)

    images_src = mv_path / "images"
    for frame in transforms["frames"]:
        src_file = mv_path / frame["file_path"]
        dst_file = images_dst / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
        frame["file_path"] = f"images/{src_file.name}"

    # Generate initial point cloud (required by OpenSplat nerfstudio format)
    points_ply = work_path / "points3d.ply"
    if not points_ply.exists():
        generate_initial_pointcloud(transforms, str(points_ply))
    transforms["ply_file_path"] = "points3d.ply"

    with open(transforms_dst, "w") as f:
        json.dump(transforms, f, indent=2)

    return str(work_path)


def generate_initial_pointcloud(transforms: dict, output_path: str, num_points: int = 5000):
    """
    Generate a random initial point cloud for 3DGS optimization.
    Distributes points in a sphere around the camera origins.
    """
    # Collect camera positions
    cam_positions = []
    for frame in transforms["frames"]:
        mat = np.array(frame["transform_matrix"])
        cam_positions.append(mat[:3, 3])
    cam_positions = np.array(cam_positions)

    # Compute bounding sphere
    center = cam_positions.mean(axis=0)
    cam_radius = np.linalg.norm(cam_positions - center, axis=1).max()
    if cam_radius < 0.01:
        cam_radius = 0.5

    # For panoramic scenes, place points on a surrounding sphere/shell
    # representing the walls/environment the cameras look at
    scene_radius = cam_radius * 5.0  # scene is further out than cameras
    rng = np.random.default_rng(42)

    # Mix of shell points (walls) and volume points (furniture/objects)
    n_shell = num_points * 3 // 4
    n_volume = num_points - n_shell

    # Shell points (on a sphere around origin)
    phi_s = rng.uniform(0, 2 * np.pi, n_shell)
    cos_theta_s = rng.uniform(-1, 1, n_shell)
    sin_theta_s = np.sqrt(1 - cos_theta_s**2)
    r_s = scene_radius * rng.uniform(0.8, 1.2, n_shell)
    shell_pts = np.column_stack([
        r_s * sin_theta_s * np.cos(phi_s),
        r_s * cos_theta_s,
        r_s * sin_theta_s * np.sin(phi_s),
    ])

    # Volume points (between cameras and walls)
    vol_pts = rng.uniform(-scene_radius * 0.6, scene_radius * 0.6, (n_volume, 3))

    points = np.vstack([shell_pts, vol_pts]) + center

    # Random colors
    colors = rng.integers(100, 200, size=(num_points, 3), dtype=np.uint8)

    # Write PLY file
    with open(output_path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for i in range(num_points):
            f.write(struct.pack("<fff", points[i, 0], points[i, 1], points[i, 2]))
            f.write(struct.pack("<BBB", colors[i, 0], colors[i, 1], colors[i, 2]))

    print(f"  Generated initial point cloud: {num_points} points → {output_path}")


def run_opensplat(
    input_dir: str,
    output_ply: str,
    num_iters: int = 30000,
    opensplat_bin: Optional[str] = None,
    extra_args: Optional[list] = None,
) -> bool:
    """Run OpenSplat reconstruction."""
    if opensplat_bin is None:
        opensplat_bin = find_opensplat()

    if opensplat_bin is None:
        print("ERROR: OpenSplat not found!")
        print("Run the setup script first:")
        print("  bash stage3_3dgs/setup_opensplat.sh")
        print("")
        print("Or install manually from: https://github.com/pierotofy/OpenSplat")
        return False

    cmd = [
        opensplat_bin,
        input_dir,
        "-o", output_ply,
        "-n", str(num_iters),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    print(f"This may take 10-30 minutes depending on your hardware...")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        for line in process.stdout:
            print(f"  [OpenSplat] {line}", end="")

        process.wait()

        if process.returncode != 0:
            print(f"\nOpenSplat exited with code {process.returncode}")
            return False

        print(f"\nReconstruction complete! Output: {output_ply}")
        return True

    except FileNotFoundError:
        print(f"ERROR: Cannot execute {opensplat_bin}")
        return False


def main():
    parser = argparse.ArgumentParser(description="3DGS reconstruction from multi-view images")
    parser.add_argument("multiview_dir", type=str, help="Directory with images/ and transforms.json")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output .ply path")
    parser.add_argument("--num-iters", type=int, default=30000, help="Training iterations")
    parser.add_argument("--opensplat", type=str, default=None, help="Path to opensplat binary")
    parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor for input images")
    parser.add_argument("--sh-degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--val", action="store_true", help="Reserve one camera for validation")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"output_dir": "outputs"}

    # Output path
    scene_name = Path(args.multiview_dir).name
    if args.output:
        output_ply = args.output
    else:
        splat_dir = Path(__file__).parent.parent / config.get("output_dir", "outputs") / "splats"
        splat_dir.mkdir(parents=True, exist_ok=True)
        output_ply = str(splat_dir / f"{scene_name}.ply")

    # Prepare workspace
    work_dir = str(Path(__file__).parent.parent / config.get("output_dir", "outputs") / "work" / scene_name)
    print(f"Preparing workspace: {work_dir}")
    prepared_dir = prepare_colmap_structure(args.multiview_dir, work_dir)

    # Build extra args
    extra_args = []
    if args.downscale > 1.0:
        extra_args.extend(["-d", str(args.downscale)])
    if args.sh_degree != 3:
        extra_args.extend(["--sh-degree", str(args.sh_degree)])
    if args.val:
        extra_args.append("--val")

    # Run reconstruction
    success = run_opensplat(
        prepared_dir,
        output_ply,
        num_iters=args.num_iters,
        opensplat_bin=args.opensplat,
        extra_args=extra_args,
    )

    if success:
        print(f"\nGaussian splat saved to: {output_ply}")
        print(f"Next step: Load in Unity with UnityGaussianSplatting plugin")
    else:
        print("\nReconstruction failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
