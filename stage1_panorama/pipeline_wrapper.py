"""
Ray Serve wrapper for Stage 1 panorama generation.

Exposes a /generate endpoint that accepts a JSON body:
  { "prompt": str, "seed": int (optional), "steps": int (optional) }
Returns: { "path": str, "clip_score": float, "qa_pass": bool }

Deploy:
    serve run stage1_panorama.pipeline_wrapper:app
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

OUTPUT_DIR = Path(os.environ.get("PANO_OUTPUT_DIR", Path.home() / "textworld" / "corpus_output"))


try:
    from ray import serve
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @serve.deployment(
        num_replicas=1,
        ray_actor_options={"num_gpus": 1},
    )
    class PanoramaGenerator:
        def __init__(self):
            from stage1_panorama.generate_panorama import load_pipeline

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.pipe = load_pipeline(self.device, dtype)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        async def __call__(self, request: Request) -> JSONResponse:
            body = await request.json()
            prompt: str = body["prompt"]
            seed: int = int(body.get("seed", 42))
            steps: int = int(body.get("steps", 40))

            safe = prompt[:50].replace(" ", "_").replace("/", "-")
            out_path = str(OUTPUT_DIR / f"{safe}_s{seed}.png")

            from stage1_panorama.generate_panorama import generate
            from stage1_panorama.panorama_qa import load_clip, qa_image

            path = generate(
                prompt=prompt,
                seed=seed,
                out_path=out_path,
                device=self.device,
                num_inference_steps=steps,
                pipe=self.pipe,
            )

            clip_state = load_clip(self.device)
            qa = qa_image(path, prompt, clip_state, self.device)

            return JSONResponse({
                "path": path,
                "clip_score": qa["clip_score"],
                "qa_pass": qa["qa_pass"],
                "pole_variance": qa["pole_variance"],
                "seam_diff": qa["seam_diff"],
            })

    app = PanoramaGenerator.bind()

except ImportError:
    class _RayNotInstalled:
        def __getattr__(self, name):
            raise ImportError("ray[serve] is required: pip install 'ray[serve]'")

    app = _RayNotInstalled()  # type: ignore
