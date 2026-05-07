#!/usr/bin/env python3
"""fill_clip.py — refill empty CLIP column in a pipeline CSV.

Reads scene ids from CSV, finds the corresponding panorama under OUTPUT_DIR/panoramas/
(matching "<prompt-slug>_s<seed>.png"), computes CLIP score, rewrites CSV.
"""
import argparse, csv, re, sys
from pathlib import Path

from evaluate import compute_clip_score


def sanitize_name(prompt: str, max_len: int = 50) -> str:
    return prompt[:max_len].replace(" ", "_").replace("/", "-")


def load_scenes_txt(scenes_txt: Path) -> dict[str, str]:
    """Map sanitized-prompt-slug -> original prompt text."""
    out = {}
    for line in scenes_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        slug = sanitize_name(line, max_len=50)
        out[slug] = line
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input pipeline CSV")
    ap.add_argument("--output-dir", required=True, help="pipeline output dir with panoramas/")
    ap.add_argument("--scenes-txt", required=True, help="scenes.txt with original prompts")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    slug2prompt = load_scenes_txt(Path(args.scenes_txt))
    print(f"loaded {len(slug2prompt)} scene prompts")

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames

    pano_dir = Path(args.output_dir) / "panoramas"
    n_ok, n_miss, n_skip = 0, 0, 0

    for row in rows:
        scene_id = row["scene"]  # e.g. "a_coffee_shop_in_tokyo_s42"
        if row.get("clip") not in (None, "", "0", "0.0"):
            n_skip += 1
            continue
        m = re.match(r"^(.*)_s(\d+)$", scene_id)
        if not m:
            print(f"  SKIP bad scene id: {scene_id}")
            n_miss += 1
            continue
        slug, seed = m.group(1), m.group(2)
        prompt = slug2prompt.get(slug)
        if prompt is None:
            for k, v in slug2prompt.items():
                if k.startswith(slug[:40]):
                    prompt = v
                    break
        if prompt is None:
            print(f"  SKIP prompt not found for slug: {slug}")
            n_miss += 1
            continue
        pano = pano_dir / f"{slug}_s{seed}.png"
        if not pano.exists():
            print(f"  SKIP missing pano: {pano.name}")
            n_miss += 1
            continue
        try:
            score = compute_clip_score(prompt, str(pano))
            row["clip"] = round(score, 4)
            row["status"] = "ok" if row.get("status", "").startswith("clip-error") else row.get("status", "ok")
            print(f"  ok  {scene_id[:55]:55s} clip={score:.4f}")
            n_ok += 1
        except Exception as e:
            print(f"  ERR {scene_id[:55]:55s} {type(e).__name__}: {e}")
            n_miss += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n== done: {n_ok} filled, {n_miss} missing, {n_skip} skipped")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
