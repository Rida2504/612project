"""Scan outputs/{panoramas,splats}/ and build the JSON triples file for
`evaluate.py compare --config`.

Each entry:
  {"pipeline": "depth-gsplat" | "legacy-v2",
   "scene":   "<prompt_stem>_s<seed>",
   "prompt":  "<full prompt text>",
   "panorama": "outputs/panoramas/<stem>.png",
   "splat":    "outputs/splats/<stem>_gsplat.ply"}

Pipeline is inferred from splat filename suffix:
  *_gsplat.ply  → depth-gsplat (our depth-init + real gsplat training)
  *.ply (no suffix) → legacy-v2 (the original random-init v2 trainer)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _sanitize(prompt: str, max_len: int = 50) -> str:
    """Mirror run_pipeline's naming: prompt[:50].replace(' ', '_').replace('/', '-')."""
    return prompt[:max_len].replace(" ", "_").replace("/", "-")


def _load_scenes(scenes_txt: Path) -> list[str]:
    prompts = []
    with open(scenes_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                        help="Root dir containing panoramas/ and splats/")
    parser.add_argument("--scenes", type=str, default="scenes.txt",
                        help="Prompt corpus file (one prompt per line, # comments allowed)")
    parser.add_argument("--out", type=str, default="-",
                        help="Output JSON path (- = stdout)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    scenes_path = Path(args.scenes)
    if not scenes_path.exists():
        # search upward relative to this file
        project_root = Path(__file__).resolve().parents[2]
        if (project_root / args.scenes).exists():
            scenes_path = project_root / args.scenes

    prompts = _load_scenes(scenes_path) if scenes_path.exists() else []
    # Build a lookup: sanitized_prefix → full prompt
    stem_to_prompt = {_sanitize(p): p for p in prompts}

    splats_dir = outputs_dir / "splats"
    panos_dir = outputs_dir / "panoramas"
    if not splats_dir.exists():
        print(f"ERROR: {splats_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    triples = []
    for ply in sorted(splats_dir.glob("*.ply")):
        name = ply.stem  # e.g. "a_cozy_library_s42_gsplat"
        if name.endswith("_gsplat"):
            pipeline = "depth-gsplat"
            scene_id = name[: -len("_gsplat")]  # "a_cozy_library_s42"
        else:
            pipeline = "legacy-v2"
            scene_id = name

        # Extract seed (look for _sNN at the end)
        m = re.match(r"^(.*)_s(\d+)$", scene_id)
        if m:
            prompt_stem = m.group(1)
        else:
            prompt_stem = scene_id

        # Look up full prompt
        full_prompt = stem_to_prompt.get(prompt_stem, prompt_stem.replace("_", " "))

        # Panorama path
        pano = panos_dir / f"{scene_id}.png"
        triples.append({
            "pipeline": pipeline,
            "scene": scene_id,
            "prompt": full_prompt,
            "panorama": str(pano) if pano.exists() else None,
            "splat": str(ply),
        })

    payload = json.dumps(triples, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).write_text(payload)
        print(f"wrote {len(triples)} triples → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
