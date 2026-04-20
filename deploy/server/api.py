"""FastAPI entry point for TextWorld VR.

Single-process, single-GPU, single-job-at-a-time. Ray Serve was the wrong
abstraction for a sequential single-GPU pipeline; this module replaces it
with plain FastAPI + uvicorn + an asyncio.Semaphore(1) that serializes jobs
on the lone GPU.

Routes:
    GET  /                  service banner
    GET  /healthz           liveness probe
    GET  /scenes            list scenes in the configured scene store
    GET  /idle              idle-watchdog probe (used by stop_when_idle.sh)
    POST /generate          enqueue a job, returns {job_id, status_url}
    GET  /status/{job_id}   poll job state
    /viewer/* and /splats/* are served by nginx, not this app.

Pipeline (executed on the GPU lock):
    Stage 1  text -> SDXL+LoRA panorama  (in-process, main env)
    Stage 2  panorama -> LP3D layered data
             (subprocesses into /opt/conda/envs/lp3d for py39 + cu118)
    Stage 3  layered data -> per-layer 3DGS -> merged .ply
             (LP3D subprocess + main-env merge tool)
    Upload   .ply -> SCENES backend (local fs or S3)
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("twvr.api")

# ---------------------------------------------------------------------------
# Paths / env
#
# APP_ROOT is the textworld-vr repo root that holds stage1_panorama/, configs/,
# etc. In the docker image this is /app; on a native VM install it is the
# checked-out repo (e.g. /mnt/src/textworld-vr). All other paths derive from
# it unless overridden.
# TOOLS_DIR holds depth_layer_fallback.py / merge_layered_plys.py — in the
# image they live in $APP_ROOT/tools (Dockerfile COPY); natively they live in
# $APP_ROOT/scripts/zaratan.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
APP_ROOT = Path(os.environ.get("APP_ROOT", _HERE.parents[2]))  # textworld-vr/
TOOLS_DIR = Path(os.environ.get("TOOLS_DIR", APP_ROOT / "scripts" / "zaratan"))
SCENES_DIR = Path(os.environ.get("SCENES_DIR", "/scenes"))
LP3D_ROOT = Path(os.environ.get("LP3D_ROOT", "/opt/LayerPano3D"))
LP3D_PY = os.environ.get("LP3D_PY", "/opt/conda/envs/lp3d/bin/python")
MAIN_PY = os.environ.get("MAIN_PY", "/opt/conda/envs/main/bin/python")
WORK_ROOT = Path(os.environ.get("WORK_DIR", "/tmp/textworld_work"))
DEFAULT_CFG = Path(os.environ.get("PIPELINE_CONFIG", APP_ROOT / "configs" / "default.yaml"))

# Make stage modules and scenes_backend importable.
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(_HERE.parent))  # deploy/server/

from scenes_backend import from_env as scenes_backend_from_env  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
SCENES = scenes_backend_from_env()  # safe at module level (no Ray pickling now)
_JOB_LOCK = asyncio.Semaphore(1)    # one GPU -> one job at a time
_JOBS: dict[str, "JobState"] = {}
_PANO_PIPE = None                   # lazy-loaded SDXL pipeline (warm across jobs)


def _pano_cfg() -> dict:
    """Panorama section of the YAML config, with deploy-time overrides applied."""
    base = yaml.safe_load(DEFAULT_CFG.read_text())
    pano = dict(base.get("panorama", {}))
    # Default config ships with device=mps (Yog's M4); override for Linux/CUDA.
    pano["device"] = os.environ.get("PANO_DEVICE", "cuda")
    pano["dtype"] = os.environ.get("PANO_DTYPE", "float16")
    return pano


def _ensure_pano_pipe():
    """Lazy-load SDXL once and keep it warm. ~30s load, ~30-60s per generation."""
    global _PANO_PIPE
    if _PANO_PIPE is None:
        log.info("loading SDXL panorama pipeline (first-job warmup)")
        from stage1_panorama.generate import load_pipeline
        _PANO_PIPE = load_pipeline(_pano_cfg())
    return _PANO_PIPE


# ---------------------------------------------------------------------------
# Pipeline stage helpers (synchronous; called from asyncio.to_thread)
# ---------------------------------------------------------------------------

def _lp3d_env_extra(extra: Optional[dict] = None) -> dict:
    """Environment vars needed by every lp3d-env subprocess.

    LP3D's pano-depth scripts depend on the 360monodepth submodule, whose
    pybind11 modules (EigenSolvers, depthmapAlign) live next to libdepth_stitch.so
    in the build tree, plus a tree of plain-Python utility/utils modules.
    The lp3d conda env provides the Ceres+glog runtime.
    """
    lp3d_lib = "/home/ubuntu/miniconda3/envs/lp3d/lib"
    pkg_parent = "/opt/LayerPano3D/submodules/360monodepth/code/cpp/python"
    pybind_dir = f"{pkg_parent}/instaOmniDepth"
    util_dir = "/opt/LayerPano3D/submodules/360monodepth/code/python/src/utility"
    src_dir = "/opt/LayerPano3D/submodules/360monodepth/code/python/src"
    pyp = ":".join([pkg_parent, pybind_dir, util_dir, src_dir, "/opt/LayerPano3D"])
    env = {
        "LD_LIBRARY_PATH": f"{pybind_dir}:{lp3d_lib}",
        "LD_PRELOAD": f"{lp3d_lib}/libglog.so.2",
        "PYTHONPATH": pyp,
    }
    if extra:
        env.update(extra)
    return env


def _run_in_env(
    python_path: str,
    args: list[str],
    cwd: Optional[str] = None,
    env_extra: Optional[dict] = None,
    timeout_s: int = 7200,
    tag: str = "",
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)
    cmd = [python_path, "-u", *args]
    log.info("[%s] exec: %s (cwd=%s)", tag, " ".join(cmd), cwd)
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, timeout=timeout_s,
        capture_output=True, text=True,
    )
    dt = time.time() - t0
    log.info(
        "[%s] exit=%d t=%.1fs stdout_bytes=%d stderr_bytes=%d",
        tag, proc.returncode, dt, len(proc.stdout), len(proc.stderr),
    )
    if proc.returncode != 0:
        log.error("[%s] stderr tail:\n%s", tag, proc.stderr[-4000:])
    return proc


def _gen_panorama(prompt: str, seed: int, out_path: str) -> str:
    """Stage 1: SDXL+LoRA -> 2048x1024 equirectangular panorama, saved to out_path."""
    from stage1_panorama.generate import build_prompt, generate_panorama
    pipe = _ensure_pano_pipe()
    cfg = _pano_cfg()
    full = build_prompt(prompt, cfg)
    img = generate_panorama(pipe, full, cfg, seed=seed)
    img.save(out_path, "PNG")
    return out_path


def _run_layering(pano_path: str, out_dir: str, n_layers: int, skip_flux: bool) -> str:
    """Stage 2: LP3D panodepth + autolayering + layerdata + (optional) FLUX-Fill traindata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(pano_path, out / "rgb.png")

    cmds: list[tuple[str, list[str], int]] = [
        ("panodepth", ["gen_panodepth.py",
                       "--input_path", str(out / "rgb.png"),
                       "--save_dir", str(out)], 1800),
        ("autolayering", ["gen_autolayering.py",
                          "--input_dir", str(out),
                          "--scene_type", "indoor"], 1200),
        ("layerdata", ["gen_layerdata.py",
                       "--base_dir", str(out / "layering")], 3600),
    ]
    if not skip_flux:
        cmds.append((
            "traindata",
            ["gen_traindata.py",
             "--layerpano_dir", str(out / "layering"),
             "--save_dir", str(out / "layering"),
             "--root", str(out / "layering")],
            3600,
        ))

    env_extra = _lp3d_env_extra({"TOKENIZERS_PARALLELISM": "false"})
    for tag, args, tmo in cmds:
        p = _run_in_env(LP3D_PY, args, cwd=str(LP3D_ROOT),
                        env_extra=env_extra, timeout_s=tmo, tag=tag)
        if p.returncode != 0:
            # If autolayering produced no instance dirs, fall back to depth-quantile splitting.
            if tag == "autolayering" and not (out / "layering" / "layer0").exists():
                log.warning("autolayering found 0 instances; running depth-quantile fallback")
                fb = _run_in_env(
                    MAIN_PY,
                    [str(TOOLS_DIR / "depth_layer_fallback.py"),
                     "--layering-dir", str(out / "layering"),
                     "--n-layers", str(n_layers)],
                    cwd=str(APP_ROOT), timeout_s=300, tag="fallback",
                )
                if fb.returncode != 0:
                    raise RuntimeError(f"fallback also failed rc={fb.returncode}")
            else:
                raise RuntimeError(f"{tag} failed rc={p.returncode}")
    return str(out / "layering")


def _run_train(layered_dir: str, out_ply: str, outlier_thresh: int = 4) -> str:
    """Stage 3: per-layer 3DGS training + merge into a single .ply."""
    scene_out = str(Path(out_ply).parent / "lp3d_scene")
    Path(scene_out).mkdir(parents=True, exist_ok=True)
    env_extra = _lp3d_env_extra()
    p = _run_in_env(
        LP3D_PY,
        ["run_layerpano.py",
         "--input_dir", layered_dir,
         "--save_dir", scene_out,
         "--outlier_thresh", str(outlier_thresh)],
        cwd=str(LP3D_ROOT), env_extra=env_extra,
        timeout_s=7200, tag="trainer",
    )
    if p.returncode != 0:
        raise RuntimeError(f"run_layerpano failed rc={p.returncode}")
    m = _run_in_env(
        MAIN_PY,
        [str(TOOLS_DIR / "merge_layered_plys.py"),
         "--scene-dir", scene_out, "--out", out_ply],
        cwd=str(APP_ROOT), timeout_s=900, tag="merge",
    )
    if m.returncode != 0:
        raise RuntimeError(f"merge failed rc={m.returncode}")
    return out_ply


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Scene description text")
    seed: int = Field(42, description="Random seed for SDXL panorama generation")
    n_layers: int = Field(3, ge=2, le=5, description="Depth-layer count")
    skip_flux: bool = Field(
        False,
        description="Skip FLUX-Fill back-layer inpainting (faster, uglier)",
    )


class JobState(BaseModel):
    job_id: str
    state: str = "queued"   # queued | panorama | layering | training | uploading | done | failed
    progress: float = 0.0
    message: str = ""
    scene_url: Optional[str] = None
    error: Optional[str] = None
    started: float
    updated: float


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="TextWorld VR", version="0.2.0")


@app.get("/")
async def root():
    return {"service": "textworld-vr", "viewer": "/viewer/", "scenes": "/scenes"}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/scenes")
async def list_scenes():
    names = SCENES.list()
    return [{"name": n, "url": SCENES.url_for(n)} for n in names]


@app.get("/idle")
async def idle():
    """Seconds since the most recent job update; used by the stop-when-idle watchdog."""
    if not _JOBS:
        # Treat "never had a job" as infinitely idle so watchdog will stop us.
        return {"last_activity_seconds_ago": 10**9, "active_jobs": 0}
    latest = max(j.updated for j in _JOBS.values())
    active = sum(1 for j in _JOBS.values() if j.state not in ("done", "failed"))
    return {"last_activity_seconds_ago": time.time() - latest, "active_jobs": active}


@app.get("/status/{job_id}")
async def status(job_id: str):
    st = _JOBS.get(job_id)
    if not st:
        raise HTTPException(404, f"unknown job {job_id}")
    return st


@app.post("/generate")
async def generate(req: GenerateRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    _JOBS[job_id] = JobState(job_id=job_id, started=now, updated=now, message="queued")
    bg.add_task(_pipeline, job_id, req)
    return {"job_id": job_id, "status_url": f"/status/{job_id}"}


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

async def _pipeline(job_id: str, req: GenerateRequest) -> None:
    """Acquire the GPU lock, run all stages, upload final .ply."""
    async with _JOB_LOCK:
        st = _JOBS[job_id]
        slug = "".join(c if c.isalnum() else "_" for c in req.prompt.strip()[:40])
        scene_name = f"{slug}_s{req.seed}"
        work = WORK_ROOT / f"{job_id}_{scene_name}"
        work.mkdir(parents=True, exist_ok=True)
        pano_path = work / "pano.png"
        layered_dir = work / "layered"
        staged_ply = work / f"{scene_name}_layered.ply"
        final_name = f"{scene_name}_layered.ply"

        try:
            st.state, st.progress, st.message, st.updated = "panorama", 0.1, "SDXL generating", time.time()
            await asyncio.to_thread(_gen_panorama, req.prompt, req.seed, str(pano_path))

            st.state, st.progress, st.message, st.updated = "layering", 0.4, "LP3D panodepth+layerdata+traindata", time.time()
            await asyncio.to_thread(_run_layering, str(pano_path), str(layered_dir),
                                    req.n_layers, req.skip_flux)

            st.state, st.progress, st.message, st.updated = "training", 0.7, "per-layer 3DGS training", time.time()
            await asyncio.to_thread(_run_train, str(layered_dir / "layering"), str(staged_ply))

            st.state, st.progress, st.message, st.updated = "uploading", 0.95, "pushing to scene store", time.time()
            await asyncio.to_thread(SCENES.put, str(staged_ply), final_name)

            st.state, st.progress, st.message, st.updated = "done", 1.0, "ok", time.time()
            st.scene_url = SCENES.url_for(final_name)
            log.info("job %s DONE -> %s", job_id, st.scene_url)
        except Exception as e:
            log.exception("job %s FAILED", job_id)
            st.state, st.error, st.message, st.updated = (
                "failed", str(e), f"pipeline error: {e}", time.time(),
            )
        finally:
            if st.state == "done":
                shutil.rmtree(work, ignore_errors=True)
            else:
                log.warning("preserving work dir for failed job: %s", work)


# ---------------------------------------------------------------------------
# Local dev (production uses uvicorn from entrypoint.sh)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
