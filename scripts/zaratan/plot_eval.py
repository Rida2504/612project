"""Read an eval CSV (from evaluate.py compare) and produce PNG plots.

Outputs:
  <out-dir>/metrics_overview.png — mean PSNR/SSIM/LPIPS/CLIP per pipeline (grouped bars)
  <out-dir>/per_scene.png        — per-scene bars of PSNR across seeds

CSV columns (from evaluate.compare_pipelines):
  pipeline,scene,clip,psnr,ssim,lpips,gaussian_count,status
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _safe_float(x):
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except (ValueError, TypeError):
        return None


def read_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in ("clip", "psnr", "ssim", "lpips", "gaussian_count"):
                if k in row:
                    row[k] = _safe_float(row[k])
            rows.append(row)
    return rows


def plot_overview(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pipelines = sorted({r["pipeline"] for r in rows})
    metric_keys = ["psnr", "ssim", "lpips", "clip"]
    # Compute mean ± std per pipeline × metric
    stats = {p: {m: [] for m in metric_keys} for p in pipelines}
    for r in rows:
        for m in metric_keys:
            v = r.get(m)
            if v is not None:
                stats[r["pipeline"]][m].append(v)

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 4))
    if len(metric_keys) == 1:
        axes = [axes]
    for ax, m in zip(axes, metric_keys):
        means, stds, labels = [], [], []
        for p in pipelines:
            vals = stats[p][m]
            labels.append(p)
            means.append(mean(vals) if vals else 0)
            stds.append(stdev(vals) if len(vals) > 1 else 0)
        xs = range(len(pipelines))
        ax.bar(xs, means, yerr=stds, capsize=4, color=["steelblue", "tomato", "gray"][: len(pipelines)])
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(m.upper())
        if m == "psnr":
            ax.set_ylabel("dB")
    fig.suptitle("TextWorld VR — metric overview by pipeline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_scene(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by scene stem (strip _sNN) to aggregate across seeds
    import re
    def _stem(s):
        m = re.match(r"^(.*)_s\d+$", s or "")
        return m.group(1) if m else (s or "unknown")

    scene_stems = sorted({_stem(r["scene"]) for r in rows})
    pipelines = sorted({r["pipeline"] for r in rows})
    psnr_per_scene = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get("psnr")
        if v is None:
            continue
        psnr_per_scene[_stem(r["scene"])][r["pipeline"]].append(v)

    fig, ax = plt.subplots(figsize=(max(8, len(scene_stems) * 1.2), 5))
    width = 0.8 / max(1, len(pipelines))
    xs = list(range(len(scene_stems)))
    for i, p in enumerate(pipelines):
        means, stds = [], []
        for s in scene_stems:
            vals = psnr_per_scene[s].get(p, [])
            means.append(mean(vals) if vals else 0)
            stds.append(stdev(vals) if len(vals) > 1 else 0)
        offsets = [x + (i - (len(pipelines) - 1) / 2) * width for x in xs]
        ax.bar(offsets, means, width=width, yerr=stds, capsize=3,
               label=p, color=["steelblue", "tomato", "gray"][i % 3])
    ax.set_xticks(xs)
    # Truncate long scene names
    labels = [s[:28] + ("…" if len(s) > 28 else "") for s in scene_stems]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-scene PSNR (mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_csv(args.csv)
    if not rows:
        print("no rows in CSV; nothing to plot")
        return
    plot_overview(rows, os.path.join(args.out_dir, "metrics_overview.png"))
    plot_per_scene(rows, os.path.join(args.out_dir, "per_scene.png"))


if __name__ == "__main__":
    main()
