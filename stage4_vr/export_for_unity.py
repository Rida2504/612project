"""
Stage 4: Prepare Gaussian Splat assets for Unity VR rendering.
Converts .ply splats into Unity-compatible format and generates
a Unity scene setup script.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def prepare_unity_assets(
    splat_ply: str,
    unity_project: str,
    scene_name: str = "GeneratedScene",
):
    """
    Copy splat .ply and metadata into a Unity project's Assets folder.

    Args:
        splat_ply: Path to the .ply Gaussian splat file
        unity_project: Path to the Unity project root
        scene_name: Name for the scene asset folder
    """
    ply_path = Path(splat_ply)
    unity_path = Path(unity_project)

    if not ply_path.exists():
        raise FileNotFoundError(f"Splat file not found: {splat_ply}")

    # Create asset directory
    assets_dir = unity_path / "Assets" / "Splats" / scene_name
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Copy .ply file
    dst_ply = assets_dir / ply_path.name
    shutil.copy2(ply_path, dst_ply)
    print(f"Copied splat: {dst_ply}")

    # Generate scene metadata
    metadata = {
        "scene_name": scene_name,
        "splat_file": str(dst_ply.relative_to(unity_path)),
        "format": "3dgs_ply",
        "rendering": {
            "target_fps": 72,
            "max_gaussians": 500000,
            "sh_degree": 3,
        },
        "vr": {
            "platform": "Meta Quest",
            "origin_position": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
    }
    meta_path = assets_dir / "scene_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_path}")

    # Generate C# scene loader script
    cs_script = generate_scene_loader(scene_name)
    scripts_dir = unity_path / "Assets" / "Scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    cs_path = scripts_dir / f"{scene_name}Loader.cs"
    with open(cs_path, "w") as f:
        f.write(cs_script)
    print(f"Scene loader script: {cs_path}")

    print(f"\nUnity assets ready in: {assets_dir}")
    print("Next steps:")
    print("  1. Open the Unity project")
    print("  2. Import UnityGaussianSplatting package")
    print(f"  3. Drag {ply_path.name} into the scene")
    print("  4. Add XR Origin for VR")
    print("  5. Build → Android → Quest")


def generate_scene_loader(scene_name: str) -> str:
    """Generate a C# MonoBehaviour for loading a splat scene in Unity."""
    return f'''using UnityEngine;

/// <summary>
/// Auto-generated scene loader for TextWorld VR scene: {scene_name}
/// Attach to an empty GameObject in your Unity scene.
/// Requires: UnityGaussianSplatting package
/// </summary>
public class {scene_name}Loader : MonoBehaviour
{{
    [Header("Splat Settings")]
    [Tooltip("Reference to the GaussianSplatRenderer component")]
    public GaussianSplatting.Runtime.GaussianSplatRenderer splatRenderer;

    [Tooltip("Scale factor for the scene")]
    public float sceneScale = 1.0f;

    [Header("VR Settings")]
    [Tooltip("Target frame rate (72 for Quest 2, 90 for Quest 3)")]
    public int targetFrameRate = 72;

    [Tooltip("Max Gaussians to render (reduce for better performance)")]
    public int maxGaussians = 500000;

    void Start()
    {{
        // Set target frame rate for Quest
        Application.targetFrameRate = targetFrameRate;

        // Set fixed timestep for smooth VR
        Time.fixedDeltaTime = 1f / targetFrameRate;

        // Apply scene scale
        if (splatRenderer != null)
        {{
            splatRenderer.transform.localScale = Vector3.one * sceneScale;
        }}

        Debug.Log($"TextWorld VR: {{name}} loaded (target {{targetFrameRate}} FPS)");
    }}

    void Update()
    {{
        // Performance monitoring
        if (Time.frameCount % 300 == 0)
        {{
            float fps = 1f / Time.deltaTime;
            if (fps < targetFrameRate * 0.9f)
            {{
                Debug.LogWarning($"FPS drop: {{fps:F1}} (target: {{targetFrameRate}})");
            }}
        }}
    }}
}}
'''


def main():
    parser = argparse.ArgumentParser(description="Export Gaussian splat for Unity VR")
    parser.add_argument("splat_ply", type=str, help="Path to .ply Gaussian splat file")
    parser.add_argument("unity_project", type=str, help="Path to Unity project root")
    parser.add_argument("--scene-name", type=str, default="GeneratedScene",
                        help="Name for the scene")
    args = parser.parse_args()

    prepare_unity_assets(args.splat_ply, args.unity_project, args.scene_name)


if __name__ == "__main__":
    main()
