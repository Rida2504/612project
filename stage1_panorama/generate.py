"""
Stage 1: Text → 360° Equirectangular Panorama Generation

Supports two modes:
  1. Full fine-tune: ProGamerGov/sdxl-360-diffusion (best quality, default)
  2. LoRA adapter: artificialguybr/360Redmond on top of SDXL base

Optimized for Apple M4 (MPS) and NVIDIA A100/H100 (CUDA).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image


def get_device(requested: str) -> torch.device:
    """Resolve the compute device."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    print(f"Warning: {requested} not available, falling back to CPU")
    return torch.device("cpu")


def load_pipeline(cfg: dict) -> DiffusionPipeline:
    """Load the 360° panorama generation pipeline."""
    device = get_device(cfg["device"])
    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.float32

    if device.type == "mps" and dtype == torch.float16:
        print("Note: Using float16 on MPS. If you see NaN errors, switch to float32 in config.")

    model_id = cfg["model_id"]
    lora_id = cfg.get("lora_id")

    # If model_id is the full fine-tune (sdxl-360-diffusion), load directly
    # If it's base SDXL + LoRA, load base then apply LoRA
    print(f"Loading model: {model_id} ...")

    # Determine if we need fp16 variant
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    # Only request fp16 variant for base SDXL (the fine-tune may not have it)
    if "stabilityai" in model_id and dtype == torch.float16:
        load_kwargs["variant"] = "fp16"

    pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)

    # Use a good scheduler for quality
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Load LoRA if specified (only for base SDXL model)
    if lora_id:
        lora_weight = cfg.get("lora_weight", 0.8)
        weight_name = cfg.get("lora_weight_name")
        print(f"Loading 360° LoRA: {lora_id} (weight={lora_weight}) ...")
        try:
            load_lora_kwargs = {}
            if weight_name:
                load_lora_kwargs["weight_name"] = weight_name
            pipe.load_lora_weights(lora_id, **load_lora_kwargs)
            pipe.fuse_lora(lora_scale=lora_weight)
            print("LoRA loaded and fused successfully.")
        except Exception as e:
            print(f"Warning: Could not load LoRA ({e}). Proceeding without it.")

    pipe = pipe.to(device)

    # Memory optimizations
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    return pipe


def build_prompt(text: str, cfg: dict) -> str:
    """Build the full prompt with panorama-specific prefix/suffix."""
    prefix = cfg.get("prompt_prefix", "")
    suffix = cfg.get("prompt_suffix", "")
    return f"{prefix}{text}{suffix}"


def generate_panorama(
    pipe: DiffusionPipeline,
    prompt: str,
    cfg: dict,
    seed: Optional[int] = None,
) -> Image.Image:
    """Generate a single 360° equirectangular panorama."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device if pipe.device.type != "mps" else "cpu")
        generator.manual_seed(seed)

    negative_prompt = cfg.get("negative_prompt", "")

    print(f"Generating {cfg['width']}x{cfg['height']} panorama ...")
    print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"  Steps: {cfg['num_inference_steps']}, CFG: {cfg['guidance_scale']}")

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=cfg["width"],
        height=cfg["height"],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
    )
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.1f}s")

    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="Generate 360° panorama from text prompt")
    parser.add_argument("prompt", type=str, help="Text description of the scene")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    parser.add_argument("--guidance", type=float, default=None, help="Override guidance scale")
    parser.add_argument("--device", type=str, default=None, help="Override device (mps/cuda/cpu)")
    parser.add_argument("--no-lora", action="store_true", help="Skip LoRA loading")
    parser.add_argument("--batch", type=int, default=1, help="Number of variations to generate")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cfg = config["panorama"]

    # CLI overrides
    if args.steps:
        cfg["num_inference_steps"] = args.steps
    if args.guidance:
        cfg["guidance_scale"] = args.guidance
    if args.device:
        cfg["device"] = args.device
    if args.no_lora:
        cfg["lora_id"] = None

    seed = args.seed if args.seed is not None else cfg.get("seed")

    # Output directory
    output_dir = Path(__file__).parent.parent / config.get("output_dir", "outputs") / "panoramas"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline(cfg)

    # Build prompt
    full_prompt = build_prompt(args.prompt, cfg)

    # Generate
    for i in range(args.batch):
        current_seed = seed + i if seed is not None else None
        image = generate_panorama(pipe, full_prompt, cfg, seed=current_seed)

        # Save
        if args.output and args.batch == 1:
            out_path = Path(args.output)
        else:
            safe_name = args.prompt[:50].replace(" ", "_").replace("/", "-")
            suffix = f"_s{current_seed}" if current_seed is not None else f"_{i}"
            out_path = output_dir / f"{safe_name}{suffix}.png"

        image.save(out_path, "PNG")
        print(f"  Saved: {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
