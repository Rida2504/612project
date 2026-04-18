"""
TextWorld VR: Evaluation Metrics

Computes:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity) — optional
  - CLIP Score (text-image alignment)
  - FID (Frechet Inception Distance) — across multiple scenes

Usage:
    # Evaluate a single scene (render quality)
    python evaluate.py render outputs/multiview/scene_name outputs/splats/scene_name.ply

    # Evaluate text-image alignment
    python evaluate.py clip "a cozy Japanese coffee shop" outputs/panoramas/scene.png

    # Compute FID across scenes
    python evaluate.py fid outputs/panoramas/ --reference-dir path/to/real/images
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ─── PSNR ─────────────────────────────────────────────────────────────────────

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images (HWC, uint8 or float32 [0,1])."""
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


# ─── SSIM ─────────────────────────────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """Compute SSIM between two images (HWC, uint8 or float32 [0,1])."""
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0

    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    # Use cv2 for Gaussian blur
    ksize = (window_size, window_size)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, ksize, sigma)
    mu2 = cv2.GaussianBlur(img2, ksize, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, ksize, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, ksize, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


# ─── LPIPS ────────────────────────────────────────────────────────────────────

_LPIPS_MODEL_CACHE = {}


def compute_lpips(img1: np.ndarray, img2: np.ndarray, net: str = "alex") -> float:
    """
    LPIPS (learned perceptual image patch similarity) — lower is better.
    Requires `pip install lpips`. Returns -1.0 on import failure.

    Inputs: HWC uint8 or float32[0,1]. Will be converted to NCHW float32[-1,1].
    """
    try:
        import lpips as _lpips
    except ImportError:
        print("Warning: lpips not installed. Run: pip install lpips")
        return -1.0

    if net not in _LPIPS_MODEL_CACHE:
        _LPIPS_MODEL_CACHE[net] = _lpips.LPIPS(net=net, verbose=False).eval()
    model = _LPIPS_MODEL_CACHE[net]

    def _prep(a: np.ndarray) -> torch.Tensor:
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        t = torch.from_numpy(a).float()
        if t.ndim == 3:
            t = t.permute(2, 0, 1).unsqueeze(0)  # NCHW
        # LPIPS expects [-1, 1]
        return t * 2.0 - 1.0

    with torch.no_grad():
        d = model(_prep(img1), _prep(img2))
    return float(d.item())


# ─── CLIP Score ───────────────────────────────────────────────────────────────

def compute_clip_score(text: str, image_path: str) -> float:
    """Compute CLIP similarity between text and image."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
    except ImportError:
        print("Warning: transformers not installed. Install with: pip install transformers")
        return -1.0

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        score = logits.item() / 100.0  # normalize to ~[0, 1]

    return float(score)


# ─── FID (simplified) ────────────────────────────────────────────────────────

def compute_fid(generated_dir: str, reference_dir: str) -> float:
    """
    Compute FID between generated and reference image directories.
    Uses InceptionV3 features (or falls back to simpler features).
    """
    try:
        from torchvision import models, transforms
        from scipy import linalg
    except ImportError:
        print("Warning: torchvision/scipy needed for FID. Install with:")
        print("  pip install torchvision scipy")
        return -1.0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load InceptionV3
    print("Loading InceptionV3 for FID...")
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()  # Remove classifier
    inception = inception.to(device).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_features(img_dir: str) -> np.ndarray:
        features = []
        img_dir = Path(img_dir)
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for img_path in sorted(img_dir.glob(ext)):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = inception(tensor).cpu().numpy().flatten()
                features.append(feat)
        return np.array(features)

    print(f"Extracting features from generated: {generated_dir}")
    feats_gen = get_features(generated_dir)
    print(f"Extracting features from reference: {reference_dir}")
    feats_ref = get_features(reference_dir)

    if len(feats_gen) < 2 or len(feats_ref) < 2:
        print("Warning: Need at least 2 images in each directory for FID")
        return -1.0

    # Compute statistics
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    mu_ref = np.mean(feats_ref, axis=0)
    sigma_ref = np.cov(feats_ref, rowvar=False)

    # FID formula
    diff = mu_gen - mu_ref
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)

    return float(fid)


# ─── Render Quality Evaluation ───────────────────────────────────────────────

def evaluate_render_quality(
    multiview_dir: str,
    splat_ply: str,
    device_str: str = "mps",
    downscale: int = 2,
) -> dict:
    """
    Evaluate 3DGS reconstruction quality by rendering from training views
    and comparing to ground truth.
    """
    from stage3_3dgs.train_3dgs_v2 import (
        GaussianModelV2, load_training_data, render_gaussians,
    )
    import struct

    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load ground truth
    images, cameras = load_training_data(multiview_dir, device, downscale)

    # Load PLY and create model
    # (simplified: we re-train or load from checkpoint)
    print("Note: Full render evaluation requires loading the trained model.")
    print("Computing metrics on saved progress renders if available...")

    progress_dir = Path(splat_ply).parent / "progress" / Path(splat_ply).stem
    if not progress_dir.exists():
        print(f"No progress renders found at {progress_dir}")
        print("Run train_3dgs_v2.py with --save-interval to generate progress renders.")
        return {}

    # Evaluate the latest progress render (left=GT, right=rendered)
    renders = sorted(progress_dir.glob("step_*.png"))
    if not renders:
        return {}

    latest = renders[-1]
    comparison = cv2.imread(str(latest))
    h, w = comparison.shape[:2]
    half_w = w // 2

    gt_img = comparison[:, :half_w]
    rendered_img = comparison[:, half_w:]

    psnr = compute_psnr(gt_img, rendered_img)
    ssim = compute_ssim(gt_img, rendered_img)

    results = {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "evaluated_render": str(latest),
        "num_training_views": len(images),
    }

    print(f"\nRender Quality Metrics:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")

    return results


# ─── Pipeline comparator ─────────────────────────────────────────────────────

def _find_progress_render(splat_ply: str) -> Optional[str]:
    """Locate the latest step_*.png under progress/{stem}/ for a given splat."""
    from pathlib import Path
    progress_dir = Path(splat_ply).parent / "progress" / Path(splat_ply).stem
    if not progress_dir.exists():
        return None
    renders = sorted(progress_dir.glob("step_*.png"))
    return str(renders[-1]) if renders else None


def evaluate_scene(
    pipeline: str,
    scene: str,
    panorama_path: Optional[str],
    splat_ply: Optional[str],
    prompt: Optional[str] = None,
    device_str: str = "mps",
    skip_clip: bool = False,
    skip_lpips: bool = False,
) -> dict:
    """Collect all automated metrics for one (pipeline, scene) pair.

    Args:
        skip_clip: skip CLIP (useful in environments where the transformers /
            TF stack crashes — e.g., macOS miniconda with protobuf conflict).
        skip_lpips: skip LPIPS if its weights can't be downloaded.
    """
    row = {
        "pipeline": pipeline,
        "scene": scene,
        "clip": None,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "gaussian_count": None,
        "status": "ok",
    }

    # CLIP on panorama
    if (not skip_clip) and prompt and panorama_path and Path(panorama_path).exists():
        try:
            row["clip"] = round(compute_clip_score(prompt, panorama_path), 4)
        except Exception as e:
            row["status"] = f"clip-error:{type(e).__name__}"
    elif skip_clip:
        row["status"] = "clip-skipped"

    # PSNR/SSIM/LPIPS from progress render comparison
    if splat_ply and Path(splat_ply).exists():
        comparison_img = _find_progress_render(splat_ply)
        if comparison_img:
            img = cv2.imread(comparison_img)
            if img is not None:
                h, w = img.shape[:2]
                half_w = w // 2
                gt = cv2.cvtColor(img[:, :half_w], cv2.COLOR_BGR2RGB)
                rendered = cv2.cvtColor(img[:, half_w:], cv2.COLOR_BGR2RGB)
                try:
                    row["psnr"] = round(compute_psnr(gt, rendered), 2)
                    row["ssim"] = round(compute_ssim(gt, rendered), 4)
                except Exception as e:
                    row["status"] = f"psnr-error:{type(e).__name__}"
                if not skip_lpips:
                    try:
                        row["lpips"] = round(compute_lpips(gt, rendered), 4)
                    except Exception as e:
                        row["status"] = f"lpips-error:{type(e).__name__}"

        # Gaussian count from .ply
        try:
            from plyfile import PlyData
            row["gaussian_count"] = int(len(PlyData.read(splat_ply)["vertex"]))
        except Exception:
            pass

    return row


def compare_pipelines(
    pipeline_scene_triples: list,
    out_csv: str,
    device_str: str = "mps",
    skip_clip: bool = False,
    skip_lpips: bool = False,
) -> list:
    """
    Run metric evaluation across multiple pipeline/scene entries, write CSV.

    Args:
        pipeline_scene_triples: list of dicts like
            {"pipeline": "depth", "scene": "japanese_coffee_shop",
             "prompt": "a cozy Japanese coffee shop",
             "panorama": "outputs/panoramas/...png",
             "splat": "outputs/splats/..._gsplat.ply"}
        out_csv: output CSV path
    """
    import csv as _csv
    rows = []
    for tri in pipeline_scene_triples:
        row = evaluate_scene(
            tri["pipeline"], tri["scene"],
            tri.get("panorama"), tri.get("splat"),
            prompt=tri.get("prompt"), device_str=device_str,
            skip_clip=skip_clip, skip_lpips=skip_lpips,
        )
        rows.append(row)
        print(f"  {tri['pipeline']}/{tri['scene']}: "
              f"clip={row['clip']} psnr={row['psnr']} ssim={row['ssim']} "
              f"lpips={row['lpips']} gauss={row['gaussian_count']}")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(
            f, fieldnames=["pipeline", "scene", "clip", "psnr", "ssim",
                           "lpips", "gaussian_count", "status"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved: {out_csv}")
    return rows


def _sanity_triples() -> list:
    """Default eval set for --sanity: uses the single baseline scene."""
    base = Path(__file__).parent
    pano = base / "outputs" / "panoramas" / "a_cozy_Japanese_coffee_shop_s42.png"
    splat_v2 = base / "outputs" / "splats" / "a_cozy_Japanese_coffee_shop_s42_v2.ply"
    prompt = "a cozy Japanese coffee shop"
    triples = []
    if pano.exists():
        triples.append({
            "pipeline": "legacy-v2",
            "scene": "japanese_coffee_shop",
            "prompt": prompt,
            "panorama": str(pano),
            "splat": str(splat_v2) if splat_v2.exists() else None,
        })
    return triples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TextWorld VR Evaluation")
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # Render quality
    render_parser = subparsers.add_parser("render", help="Evaluate render quality (PSNR, SSIM)")
    render_parser.add_argument("multiview_dir", type=str)
    render_parser.add_argument("splat_ply", type=str)
    render_parser.add_argument("--device", type=str, default="mps")
    render_parser.add_argument("--downscale", type=int, default=2)

    # CLIP score
    clip_parser = subparsers.add_parser("clip", help="Evaluate text-image alignment")
    clip_parser.add_argument("text", type=str)
    clip_parser.add_argument("image", type=str)

    # FID
    fid_parser = subparsers.add_parser("fid", help="Compute FID score")
    fid_parser.add_argument("generated_dir", type=str)
    fid_parser.add_argument("--reference-dir", type=str, required=True)

    # Compare pipelines
    cmp_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple pipelines × scenes. Produces a CSV of metrics.",
    )
    cmp_parser.add_argument("--config", type=str, default=None,
                            help="JSON file of [{pipeline, scene, prompt, panorama, splat}, ...]")
    cmp_parser.add_argument("--out", type=str, default="outputs/eval_compare.csv")
    cmp_parser.add_argument("--sanity", action="store_true",
                            help="Run on the built-in single-scene baseline (smoke test)")
    cmp_parser.add_argument("--skip-clip", action="store_true",
                            help="Skip CLIP (use on envs where transformers/TF crashes)")
    cmp_parser.add_argument("--skip-lpips", action="store_true",
                            help="Skip LPIPS (use if weights can't download)")
    cmp_parser.add_argument("--device", type=str, default="mps")

    # All metrics for a scene
    all_parser = subparsers.add_parser("all", help="Run all metrics for a scene")
    all_parser.add_argument("--prompt", type=str, required=True)
    all_parser.add_argument("--panorama", type=str, required=True)
    all_parser.add_argument("--multiview-dir", type=str, required=True)
    all_parser.add_argument("--splat-ply", type=str, required=True)
    all_parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()

    if args.mode == "render":
        results = evaluate_render_quality(
            args.multiview_dir, args.splat_ply,
            device_str=args.device, downscale=args.downscale,
        )
        if results:
            out_path = Path(args.splat_ply).with_name("eval_render.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved: {out_path}")

    elif args.mode == "clip":
        score = compute_clip_score(args.text, args.image)
        print(f"\nCLIP Score: {score:.4f}")

    elif args.mode == "fid":
        fid = compute_fid(args.generated_dir, args.reference_dir)
        print(f"\nFID Score: {fid:.2f}")

    elif args.mode == "compare":
        if args.sanity:
            triples = _sanity_triples()
            if not triples:
                # Still emit a header-only CSV so downstream tooling can load it
                print("[compare] No sanity triples found (missing outputs); writing header-only CSV.")
                triples = []
        elif args.config:
            with open(args.config) as f:
                triples = json.load(f)
        else:
            print("ERROR: pass --sanity or --config")
            return
        compare_pipelines(
            triples, args.out, device_str=args.device,
            skip_clip=args.skip_clip, skip_lpips=args.skip_lpips,
        )

    elif args.mode == "all":
        print("=" * 60)
        print("TextWorld VR: Full Evaluation")
        print("=" * 60)

        results = {}

        # CLIP score
        print("\n--- CLIP Score ---")
        clip = compute_clip_score(args.prompt, args.panorama)
        results["clip_score"] = round(clip, 4)
        print(f"CLIP: {clip:.4f}")

        # Render quality
        print("\n--- Render Quality ---")
        render_metrics = evaluate_render_quality(
            args.multiview_dir, args.splat_ply, device_str=args.device,
        )
        results.update(render_metrics)

        results["prompt"] = args.prompt
        results["panorama"] = args.panorama

        out_path = Path(args.splat_ply).with_name("eval_full.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
