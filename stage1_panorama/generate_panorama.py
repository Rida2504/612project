"""
Stage 1: Text → Equirectangular Panorama

Generates a 2048×1024 equirectangular panorama PNG from a text prompt using
SDXL-base with an optional panoramic LoRA.

Usage:
    python generate_panorama.py \
        --prompt "a cozy Japanese coffee shop with warm Edison lighting" \
        --seed 42 \
        --out ~/textworld/corpus_output/test.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# xformers removed — incompatible with torch 2.5.1 on CUDA 13
# pipe.enable_xformers_memory_efficient_attention()

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Local LoRA (provided by Yog). Falls back to HF LoRA if not present.
LOCAL_LORA_PATH = Path.home() / "textworld" / "lora_hubs" / "pano_lora_720x1440_v1.safetensors"
HF_LORA_ID = "artificialguybr/360Redmond"
HF_LORA_WEIGHT_NAME = "View360.safetensors"

PROMPT_PREFIX = "360 panoramic view of "
PROMPT_SUFFIX = ", equirectangular projection, high quality, detailed, photorealistic, seamless"
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, text, logo, seam visible, "
    "low resolution, cartoon, painting, perspective distortion"
)

OUTPUT_W = 2048
OUTPUT_H = 1024


def blend_seam(img: np.ndarray, blend_width: int = 64) -> np.ndarray:
    """Linearly blend left and right edges to reduce the vertical wraparound seam."""
    w = img.shape[1]
    left = img[:, :blend_width].astype(np.float32)
    right = img[:, w - blend_width:].astype(np.float32)
    alpha = np.linspace(0, 1, blend_width)[None, :, None]
    blended = (1 - alpha) * right + alpha * left
    out = img.copy()
    out[:, :blend_width] = blended.astype(np.uint8)
    out[:, w - blend_width:] = blended.astype(np.uint8)
    return out


def load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if LOCAL_LORA_PATH.exists():
        print(f"Loading local LoRA: {LOCAL_LORA_PATH}")
        pipe.load_lora_weights(str(LOCAL_LORA_PATH.parent), weight_name=LOCAL_LORA_PATH.name)
    else:
        print(f"Local LoRA not found at {LOCAL_LORA_PATH} — loading HF fallback: {HF_LORA_ID}")
        pipe.load_lora_weights(HF_LORA_ID, weight_name=HF_LORA_WEIGHT_NAME)

    return pipe


def generate(
    prompt: str,
    seed: int = 42,
    out_path: str | None = None,
    device: str = "cuda",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    blend_seam_width: int = 64,
    pipe: StableDiffusionXLPipeline | None = None,
) -> str:
    """Generate one panorama. Returns the output file path."""
    if pipe is None:
        dtype = torch.float16 if device != "cpu" else torch.float32
        pipe = load_pipeline(device, dtype)

    full_prompt = PROMPT_PREFIX + prompt + PROMPT_SUFFIX
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=OUTPUT_W,
        height=OUTPUT_H,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    img = result.images[0]
    img_np = np.array(img)
    if blend_seam_width > 0:
        img_np = blend_seam(img_np, blend_seam_width)
    img = Image.fromarray(img_np)

    if out_path is None:
        safe = prompt[:50].replace(" ", "_").replace("/", "-")
        out_path = str(Path("outputs") / "panoramas" / f"{safe}_s{seed}.png")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate a 360° equirectangular panorama")
    parser.add_argument("--prompt", type=str, required=True, help="Scene description")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--no-blend", action="store_true", help="Skip seam blending")
    args = parser.parse_args()

    generate(
        prompt=args.prompt,
        seed=args.seed,
        out_path=args.out,
        device=args.device,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        blend_seam_width=0 if args.no_blend else 64,
    )


if __name__ == "__main__":
    main()
