"""
Panorama QA: CLIP relevance + pole artifact + seam artifact checks.

Thresholds are relaxed relative to ideal LoRA-tuned outputs because
SDXL without the custom LoRA produces noisier pole regions and seams.
Will tighten once pano_lora_720x1440_v1.safetensors is loaded.

Usage:
    python panorama_qa.py image.png --prompt "scene description"
    python panorama_qa.py *.png --prompts-csv batch_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# CLIP thresholds
CLIP_THRESHOLD = 0.28

# Pole check: measure variance in the top/bottom 5% of rows.
# High variance → pole artifact. Relaxed from 150 → 500 (no LoRA baseline).
POLE_VARIANCE_THRESHOLD = 500

# Seam check: measure absolute difference between leftmost and rightmost columns.
# High diff → visible seam. Relaxed from 40 → 60 (no LoRA baseline).
SEAM_DIFF_THRESHOLD = 60


def load_clip(device: str = "cuda"):
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model = model.to(device).eval()
        return model, preprocess, tokenizer, device
    except ImportError:
        raise ImportError("open-clip-torch is required: pip install open-clip-torch")


def clip_score(img: Image.Image, prompt: str, clip_state) -> float:
    model, preprocess, tokenizer, device = clip_state
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    text_tokens = tokenizer([prompt]).to(device)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        txt_features = model.encode_text(text_tokens)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        txt_features /= txt_features.norm(dim=-1, keepdim=True)
        score = (img_features @ txt_features.T).item()

    return float(score)


def pole_check(img_np: np.ndarray) -> tuple[bool, float]:
    """Check top and bottom pole regions for artifact variance."""
    h = img_np.shape[0]
    pole_h = max(1, int(h * 0.05))
    top_var = float(np.var(img_np[:pole_h].astype(np.float32)))
    bot_var = float(np.var(img_np[-pole_h:].astype(np.float32)))
    max_var = max(top_var, bot_var)
    return max_var <= POLE_VARIANCE_THRESHOLD, max_var


def seam_check(img_np: np.ndarray) -> tuple[bool, float]:
    """Check left/right edge difference for wraparound seam visibility."""
    left_col = img_np[:, 0].astype(np.float32)
    right_col = img_np[:, -1].astype(np.float32)
    diff = float(np.mean(np.abs(left_col - right_col)))
    return diff <= SEAM_DIFF_THRESHOLD, diff


def qa_image(
    image_path: str,
    prompt: str,
    clip_state=None,
    device: str = "cuda",
) -> dict:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    if clip_state is None:
        clip_state = load_clip(device)

    cs = clip_score(img, prompt, clip_state)
    pole_ok, pole_val = pole_check(img_np)
    seam_ok, seam_val = seam_check(img_np)

    passed = cs >= CLIP_THRESHOLD and pole_ok and seam_ok

    return {
        "path": str(image_path),
        "prompt": prompt,
        "clip_score": round(cs, 4),
        "clip_pass": cs >= CLIP_THRESHOLD,
        "pole_variance": round(pole_val, 1),
        "pole_pass": pole_ok,
        "seam_diff": round(seam_val, 2),
        "seam_pass": seam_ok,
        "qa_pass": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="QA check for panorama images")
    parser.add_argument("images", nargs="+", help="PNG file paths")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for CLIP scoring")
    parser.add_argument(
        "--prompts-csv",
        type=str,
        default=None,
        help="CSV with columns: path,prompt (for batch mode)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-csv", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    clip_state = load_clip(args.device)

    pairs: list[tuple[str, str]] = []
    if args.prompts_csv:
        with open(args.prompts_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["path"], row["prompt"]))
    else:
        if args.prompt is None:
            print("ERROR: --prompt required when not using --prompts-csv")
            sys.exit(1)
        for img in args.images:
            pairs.append((img, args.prompt))

    results = []
    for path, prompt in pairs:
        r = qa_image(path, prompt, clip_state, args.device)
        status = "PASS" if r["qa_pass"] else "FAIL"
        print(
            f"[{status}] {Path(path).name} | CLIP={r['clip_score']:.4f} "
            f"| pole_var={r['pole_variance']} | seam_diff={r['seam_diff']}"
        )
        results.append(r)

    n_pass = sum(1 for r in results if r["qa_pass"])
    print(f"\n{n_pass}/{len(results)} passed QA")

    if args.out_csv:
        fieldnames = list(results[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved: {args.out_csv}")


if __name__ == "__main__":
    main()
