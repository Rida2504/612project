"""
stage_models.py — idempotent download of every model the pipeline needs.
Runs on container boot. Skips anything already cached.

Required (pulled to $HF_HOME / hf_cache):
  - stabilityai/stable-diffusion-xl-base-1.0        (SDXL)
  - black-forest-labs/FLUX.1-dev                    (gated; needs HF_TOKEN accepted)
  - black-forest-labs/FLUX.1-Fill-dev               (gated)
  - ysmikey/Layerpano3D-FLUX-Panorama-LoRA          (LoRA)
  - Johanan0528/Infusion                            (depth-inpaint SD)
  - shi-labs/oneformer_coco_swin_large              (panoptic seg)
  - llava-hf/llava-1.5-7b-hf                        (caption)
  - depth-anything/Depth-Anything-V2-Large          (DA-v2 ViT-L checkpoint)

Required as direct URLs (pulled to $LP3D_CHECKPOINTS):
  - sam_vit_h_4b8939.pth                            (SAM)
  - big-lama.zip                                    (LaMa)
  - pano_lora_720*1440_v1.safetensors               (symlink from HF LoRA cache)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

HF_SNAPSHOT_REPOS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", None),
    ("black-forest-labs/FLUX.1-dev", None),
    ("black-forest-labs/FLUX.1-Fill-dev", None),
    ("ysmikey/Layerpano3D-FLUX-Panorama-LoRA",
     ["lora_hubs/pano_lora_720*1440_v1.safetensors"]),
    ("Johanan0528/Infusion", None),
    ("shi-labs/oneformer_coco_swin_large", None),
    ("llava-hf/llava-1.5-7b-hf", None),
]

DIRECT_URLS = [
    ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
     "sam_vit_h_4b8939.pth"),
    ("https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip",
     "big-lama.zip"),
    # DA-v2 vitl checkpoint — needed for gen_panodepth, gen_autolayering, gen_traindata
    ("https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
     "depth_anything_v2_vitl.pth"),
    # ControlNet-style repackaged LaMa weights — gen_layerdata.py loads
    # checkpoints/ControlNetLama.pth via utils.lama.LamaInpainting.
    ("https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth",
     "ControlNetLama.pth"),
]


def stage_hf(repo: str, allow_patterns, cache_dir: str, token: str | None) -> Path | None:
    from huggingface_hub import snapshot_download
    try:
        print(f"[models] {repo}: snapshot_download start", flush=True)
        p = snapshot_download(repo_id=repo, cache_dir=cache_dir, token=token,
                              allow_patterns=allow_patterns)
        print(f"[models] {repo}: ok -> {p}", flush=True)
        return Path(p)
    except Exception as e:
        print(f"[models] {repo}: FAIL {e}", file=sys.stderr, flush=True)
        raise


def stage_url(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[models] skip-exists {dest.name} ({dest.stat().st_size // 1_000_000} MB)")
        return
    print(f"[models] download {url} -> {dest}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    tmp.rename(dest)
    print(f"[models] ok {dest} ({dest.stat().st_size // 1_000_000} MB)")


def _resolve_snapshot(repo: str, cache_dir: Path, token: str | None) -> Path | None:
    """Return the snapshot dir for `repo`, regardless of which HF cache layout is in use.

    huggingface_hub uses two layouts depending on how the cache was selected:
      - cache_dir passed explicitly to snapshot_download → `<cache_dir>/models--<org>--<name>/snapshots/<sha>/`
      - HF_HOME / default → `<HF_HOME>/hub/models--<org>--<name>/snapshots/<sha>/`
    Try the explicit-cache layout first, then the HF_HOME `hub/` layout, then fall back
    to re-calling `snapshot_download` (idempotent — returns the local path immediately
    when fully cached).
    """
    repo_dirname = "models--" + repo.replace("/", "--")
    for base in (cache_dir, cache_dir / "hub"):
        snapshots_dir = base / repo_dirname / "snapshots"
        if snapshots_dir.is_dir():
            for snap in snapshots_dir.iterdir():
                if snap.is_dir():
                    return snap
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=repo, cache_dir=str(cache_dir), token=token))
    except Exception as e:
        print(f"[models] cannot resolve snapshot for {repo}: {e}", file=sys.stderr)
        return None


def link_lora(snap: Path | None, checkpoints: Path) -> None:
    """Symlink the literal-asterisk LoRA file into checkpoints/ so LP3D scripts find it."""
    if snap is None:
        print("[models] no LoRA snapshot cached yet", file=sys.stderr)
        return
    fname = "pano_lora_720*1440_v1.safetensors"
    src = snap / "lora_hubs" / fname
    if not src.exists():
        candidates = list((snap / "lora_hubs").glob(fname)) if (snap / "lora_hubs").is_dir() else []
        if not candidates:
            print(f"[models] missing LoRA file under {snap}", file=sys.stderr)
            return
        src = candidates[0]
    dst = checkpoints / fname
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)
    print(f"[models] linked LoRA -> {dst}")


def link_infusion(snap: Path | None, checkpoints: Path) -> None:
    if snap is None:
        print("[models] no Infusion snapshot cached yet", file=sys.stderr)
        return
    dst = checkpoints / "Infusion"
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        return
    dst.symlink_to(snap)
    print(f"[models] linked Infusion -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HOME", "/models/hf_cache"))
    ap.add_argument("--checkpoints", default=os.environ.get("LP3D_CHECKPOINTS", "/opt/LayerPano3D/checkpoints"))
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("[models] WARNING: HF_TOKEN not set — gated models (FLUX) will fail", file=sys.stderr)

    cache_dir = Path(args.hf_cache)
    checkpoints = Path(args.checkpoints)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    snapshots: dict[str, Path | None] = {}
    for repo, patterns in HF_SNAPSHOT_REPOS:
        try:
            snapshots[repo] = stage_hf(repo, patterns, args.hf_cache, token)
        except Exception:
            # Non-fatal — user may pre-seed caches via volume mount; try to resolve below.
            snapshots[repo] = None

    for url, fname in DIRECT_URLS:
        stage_url(url, checkpoints / fname)

    lora_snap = snapshots.get("ysmikey/Layerpano3D-FLUX-Panorama-LoRA") \
        or _resolve_snapshot("ysmikey/Layerpano3D-FLUX-Panorama-LoRA", cache_dir, token)
    infusion_snap = snapshots.get("Johanan0528/Infusion") \
        or _resolve_snapshot("Johanan0528/Infusion", cache_dir, token)

    link_lora(lora_snap, checkpoints)
    link_infusion(infusion_snap, checkpoints)

    print("[models] staging complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
