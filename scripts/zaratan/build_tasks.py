"""Emit a tasks.tsv (prompt<TAB>seed, one per line) from scenes.txt × seeds."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default="scenes.txt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out", default="tasks.tsv")
    args = parser.parse_args()

    scenes_path = Path(args.scenes)
    if not scenes_path.exists():
        scenes_path = Path(__file__).resolve().parents[2] / args.scenes
    prompts = []
    with open(scenes_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)

    if not prompts:
        print("ERROR: no prompts in scenes.txt", file=sys.stderr)
        sys.exit(1)

    lines = []
    for seed in args.seeds:
        for prompt in prompts:
            lines.append(f"{prompt}\t{seed}")

    Path(args.out).write_text("\n".join(lines) + "\n")
<<<<<<< HEAD
    print(f"wrote {len(lines)} tasks ({len(prompts)} prompts × {len(args.seeds)} seeds) → {args.out}")
=======
    print(
        f"wrote {len(lines)} tasks ({len(prompts)} prompts × {len(args.seeds)} seeds) → {args.out}"
    )
>>>>>>> main


if __name__ == "__main__":
    main()
