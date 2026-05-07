"""
eval_novel_view.py — render N novel views from a .ply and compute CLIP score
against the prompt. Prints per-view scores + mean. Optional view-consistency
LPIPS between first view (origin) and the rest.

This is the honest evaluation the v2 pipeline failed: novel-view CLIP matches
the semantics the user actually asked for, whereas SGD-memorized training-view
PSNR does not.

Usage:
  python eval_novel_view.py --ply <.ply> --prompt "a cozy kitchen..." \
      --n-views 8 --radius 0.3 [--out /tmp/eval.json]
"""
<<<<<<< HEAD
=======

>>>>>>> main
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Re-use the ply loader / camera builder from parallax_proof.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallax_proof import load_ply_to_gs, camera_at  # noqa: E402


<<<<<<< HEAD
def render_view(gs: dict, cam_pos, look_at, width: int, height: int,
                fov_deg: float = 90.0) -> np.ndarray:
    from gsplat.rendering import rasterization
=======
def render_view(
    gs: dict, cam_pos, look_at, width: int, height: int, fov_deg: float = 90.0
) -> np.ndarray:
    from gsplat.rendering import rasterization

>>>>>>> main
    device = gs["means"].device
    view = camera_at(cam_pos, look_at=look_at, device=str(device))
    fx = width / (2 * math.tan(math.radians(fov_deg) / 2))
    fy = height / (2 * math.tan(math.radians(fov_deg) / 2))
<<<<<<< HEAD
    K = torch.tensor([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]],
                     dtype=torch.float32, device=device).unsqueeze(0)
    sh_degree = int(math.isqrt(gs["sh"].shape[1])) - 1
    imgs, _, _ = rasterization(
        means=gs["means"], quats=gs["quats"], scales=gs["scales"],
        opacities=gs["opacities"], colors=gs["sh"],
        viewmats=view, Ks=K, width=width, height=height,
=======
    K = torch.tensor(
        [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    sh_degree = int(math.isqrt(gs["sh"].shape[1])) - 1
    imgs, _, _ = rasterization(
        means=gs["means"],
        quats=gs["quats"],
        scales=gs["scales"],
        opacities=gs["opacities"],
        colors=gs["sh"],
        viewmats=view,
        Ks=K,
        width=width,
        height=height,
>>>>>>> main
        sh_degree=max(0, sh_degree),
    )
    return (imgs[0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)


def generate_camera_poses(n: int, radius: float):
    """Generate N camera positions in a horizontal circle around origin."""
    poses = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)  # stay in the XZ plane
        # Look tangentially forward so the scene is in-frame
        look_x = x + 10 * math.cos(theta + math.pi / 2)
        look_z = z + 10 * math.sin(theta + math.pi / 2)
        poses.append(((x, 0, z), (look_x, 0, look_z)))
    return poses


def clip_score(img_np_list, prompt: str, device: str = "cuda") -> list[float]:
    """Return CLIP cosine similarity per image vs prompt."""
    import open_clip
    from PIL import Image
<<<<<<< HEAD
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32",
        pretrained="laion2b_s34b_b79k", device=device)
=======

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
>>>>>>> main
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    with torch.no_grad():
        text = tok([prompt]).to(device)
        txt_feat = model.encode_text(text)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        scores = []
        for img in img_np_list:
            pil = Image.fromarray(img)
            x = preprocess(pil).unsqueeze(0).to(device)
            feat = model.encode_image(x)
            feat /= feat.norm(dim=-1, keepdim=True)
            scores.append(float((feat @ txt_feat.T).squeeze().cpu()))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n-views", type=int, default=8)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", default=None)
<<<<<<< HEAD
    ap.add_argument("--save-frames-dir", default=None,
                    help="if set, dump rendered PNGs to this directory")
=======
    ap.add_argument(
        "--save-frames-dir",
        default=None,
        help="if set, dump rendered PNGs to this directory",
    )
>>>>>>> main
    args = ap.parse_args()

    gs = load_ply_to_gs(args.ply, device="cuda")
    print(f"[eval] loaded {args.ply} — {gs['means'].shape[0]} points")
    poses = generate_camera_poses(args.n_views, args.radius)
    frames = []
    for i, (pos, look) in enumerate(poses):
        img = render_view(gs, pos, look, args.width, args.height)
        frames.append(img)
        if args.save_frames_dir:
            from PIL import Image
<<<<<<< HEAD
            os.makedirs(args.save_frames_dir, exist_ok=True)
            Image.fromarray(img).save(os.path.join(args.save_frames_dir, f"view_{i}.png"))
=======

            os.makedirs(args.save_frames_dir, exist_ok=True)
            Image.fromarray(img).save(
                os.path.join(args.save_frames_dir, f"view_{i}.png")
            )
>>>>>>> main
    del gs
    torch.cuda.empty_cache()

    scores = clip_score(frames, args.prompt)
    print(f"[eval] per-view CLIP: {[f'{s:.3f}' for s in scores]}")
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"[eval] mean_clip={mean:.4f}  std={std:.4f}")
    result = {
        "ply": args.ply,
        "prompt": args.prompt,
        "n_views": args.n_views,
        "radius": args.radius,
        "per_view_clip": scores,
        "mean_clip": mean,
        "std_clip": std,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] wrote {args.out}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
