"""Pixel Duet Web API — FastAPI application with background job processing."""

import asyncio
import contextlib
import os
import tempfile
import threading
import time
import urllib.request
import uuid
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pixelduet.animation import EASINGS

# Job TTL: remove completed/failed jobs older than this (seconds)
_JOB_TTL = 3600  # 1 hour

# ---------- Presets ----------

PRESETS = [
    {
        "id": "smooth",
        "name": "Smooth Flow",
        "params": {"stagger": 0.5, "arc": 0.05, "easing": "cosine"},
    },
    {
        "id": "explosive",
        "name": "Explosive",
        "params": {"stagger": 0.1, "arc": 0.25, "easing": "expo"},
    },
    {
        "id": "bounce",
        "name": "Bouncy",
        "params": {"stagger": 0.3, "arc": 0.15, "easing": "bounce"},
    },
    {
        "id": "elastic",
        "name": "Elastic Snap",
        "params": {"stagger": 0.2, "arc": 0.12, "easing": "elastic"},
    },
    {
        "id": "pixelrain",
        "name": "Pixel Rain",
        "params": {"stagger": 0.8, "arc": 0.02, "easing": "linear"},
    },
    {
        "id": "staccato",
        "name": "Staccato",
        "params": {"stagger": 0.4, "arc": 0.08, "easing": "step"},
    },
]


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class Job:
    """In-memory representation of a processing job."""

    __slots__ = (
        "created_at",
        "error",
        "format",
        "id",
        "output_path",
        "progress",
        "stage",
        "status",
    )

    def __init__(self, job_id, fmt="gif"):
        self.id = job_id
        self.status = JobStatus.pending
        self.stage = ""
        self.progress = 0.0
        self.output_path = None
        self.error = None
        self.created_at = time.time()
        self.format = fmt


# In-memory job store (suitable for single-process deployment)
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

# Temp directory for uploads and outputs
_WORK_DIR = Path(tempfile.mkdtemp(prefix="pixelduet_"))


def _get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _cleanup_old_jobs():
    """Remove expired jobs and their output files."""
    now = time.time()
    expired = []
    with _jobs_lock:
        for jid, job in _jobs.items():
            if job.status in (JobStatus.done, JobStatus.failed) and now - job.created_at > _JOB_TTL:
                expired.append(jid)
        for jid in expired:
            job = _jobs.pop(jid, None)
            if job and job.output_path and os.path.exists(job.output_path):
                with contextlib.suppress(OSError):
                    os.remove(job.output_path)


def _run_job(job: Job, path_a: str, path_b: str, params: dict):
    """Execute a pixel-duet job in a background thread."""
    try:
        job.status = JobStatus.running
        job.stage = "importing"

        from pixelduet import pixel_duet

        ext = params.get("format", "gif")
        output_path = str(_WORK_DIR / f"{job.id}.{ext}")

        def on_progress(stage, frac):
            job.stage = stage
            job.progress = round(frac * 100, 1)

        pixel_duet(
            path_a,
            path_b,
            output_path=output_path,
            width=params.get("width", 260),
            frames=params.get("frames", 120),
            hold=params.get("hold", 24),
            stagger=params.get("stagger", 0.35),
            arc=params.get("arc", 0.10),
            seed=params.get("seed"),
            fps=params.get("fps", 30),
            easing=params.get("easing", "cosine"),
            progress_callback=on_progress,
        )

        job.output_path = output_path
        job.status = JobStatus.done
        job.progress = 100.0
        job.stage = "done"
    except Exception as exc:
        job.status = JobStatus.failed
        job.error = str(exc)
        job.stage = "failed"


def _start_job(job: Job, path_a: str, path_b: str, params: dict):
    """Register a job and launch its background thread."""
    with _jobs_lock:
        _jobs[job.id] = job
    thread = threading.Thread(target=_run_job, args=(job, path_a, path_b, params), daemon=True)
    thread.start()


def create_app(static_dir: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Pixel Duet", version="0.1.0")

    # ---------- Presets ----------

    @app.get("/api/presets")
    async def get_presets():
        """Return available animation presets."""
        return PRESETS

    # ---------- Submit job (file upload) ----------

    @app.post("/api/duet")
    async def submit_job(
        image_a: UploadFile = File(...),  # noqa: B008
        image_b: UploadFile = File(...),  # noqa: B008
        width: int = Form(260),
        frames: int = Form(120),
        hold: int = Form(24),
        stagger: float = Form(0.35),
        arc: float = Form(0.10),
        fps: int = Form(30),
        format: str = Form("gif"),
        easing: str = Form("cosine"),
    ):
        """Upload two images and start a pixel-duet job. Returns a job ID."""
        _cleanup_old_jobs()

        if format not in ("gif", "mp4", "webm"):
            return JSONResponse({"error": f"Unsupported format: {format}"}, status_code=400)
        if easing not in EASINGS:
            return JSONResponse(
                {"error": f"Unknown easing '{easing}'. Available: {', '.join(EASINGS)}"},
                status_code=400,
            )

        job_id = uuid.uuid4().hex[:12]

        # Save uploads to temp files
        path_a = str(_WORK_DIR / f"{job_id}_a{Path(image_a.filename or 'a.png').suffix}")
        path_b = str(_WORK_DIR / f"{job_id}_b{Path(image_b.filename or 'b.png').suffix}")

        content_a = await image_a.read()
        content_b = await image_b.read()
        with open(path_a, "wb") as f:
            f.write(content_a)
        with open(path_b, "wb") as f:
            f.write(content_b)

        job = Job(job_id, fmt=format)
        params = {
            "width": width,
            "frames": frames,
            "hold": hold,
            "stagger": stagger,
            "arc": arc,
            "fps": fps,
            "format": format,
            "easing": easing,
        }

        _start_job(job, path_a, path_b, params)
        return {"job_id": job_id}

    # ---------- Submit job (URL input) ----------

    @app.post("/api/duet/url")
    async def submit_job_url(
        image_a_url: str = Form(...),
        image_b_url: str = Form(...),
        width: int = Form(260),
        frames: int = Form(120),
        hold: int = Form(24),
        stagger: float = Form(0.35),
        arc: float = Form(0.10),
        fps: int = Form(30),
        format: str = Form("gif"),
        easing: str = Form("cosine"),
    ):
        """Accept image URLs, download them, and start a pixel-duet job."""
        _cleanup_old_jobs()

        if format not in ("gif", "mp4", "webm"):
            return JSONResponse({"error": f"Unsupported format: {format}"}, status_code=400)
        if easing not in EASINGS:
            return JSONResponse(
                {"error": f"Unknown easing '{easing}'. Available: {', '.join(EASINGS)}"},
                status_code=400,
            )

        job_id = uuid.uuid4().hex[:12]

        # Download images
        try:
            path_a = str(_WORK_DIR / f"{job_id}_a.png")
            path_b = str(_WORK_DIR / f"{job_id}_b.png")
            urllib.request.urlretrieve(image_a_url, path_a)
            urllib.request.urlretrieve(image_b_url, path_b)
        except Exception as exc:
            return JSONResponse({"error": f"Failed to download images: {exc}"}, status_code=400)

        job = Job(job_id, fmt=format)
        params = {
            "width": width,
            "frames": frames,
            "hold": hold,
            "stagger": stagger,
            "arc": arc,
            "fps": fps,
            "format": format,
            "easing": easing,
        }

        _start_job(job, path_a, path_b, params)
        return {"job_id": job_id}

    # ---------- Job status ----------

    @app.get("/api/jobs/{job_id}/status")
    async def job_status(job_id: str):
        """Check the status of a job."""
        job = _get_job(job_id)
        if job is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "stage": job.stage,
            "progress": job.progress,
            "error": job.error,
        }

    # ---------- Job result ----------

    @app.get("/api/jobs/{job_id}/result")
    async def job_result(job_id: str):
        """Download the result file for a completed job."""
        job = _get_job(job_id)
        if job is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        if job.status != JobStatus.done:
            return JSONResponse(
                {"error": f"Job is {job.status.value}, not done"},
                status_code=409,
            )
        if not job.output_path or not os.path.exists(job.output_path):
            return JSONResponse({"error": "Output file missing"}, status_code=500)

        ext = Path(job.output_path).suffix.lstrip(".")
        media_types = {"gif": "image/gif", "mp4": "video/mp4", "webm": "video/webm"}
        return FileResponse(
            job.output_path,
            media_type=media_types.get(ext, "application/octet-stream"),
            filename=f"pixel_duet.{ext}",
        )

    # ---------- Share link ----------

    @app.get("/api/jobs/{job_id}/share")
    async def job_share(job_id: str):
        """Return a shareable link for a completed job."""
        job = _get_job(job_id)
        if job is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        if job.status != JobStatus.done:
            return JSONResponse(
                {"error": f"Job is {job.status.value}, not done"},
                status_code=409,
            )
        remaining = max(0, _JOB_TTL - (time.time() - job.created_at))
        return {
            "url": f"/api/jobs/{job_id}/result",
            "job_id": job_id,
            "expires_in": round(remaining),
        }

    # ---------- Gallery ----------

    @app.get("/api/gallery")
    async def gallery():
        """Return a list of completed jobs, newest first, max 50."""
        with _jobs_lock:
            completed = [
                {
                    "job_id": job.id,
                    "created_at": job.created_at,
                    "format": job.format,
                    "status": job.status.value,
                }
                for job in sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
                if job.status == JobStatus.done
            ][:50]
        return completed

    # ---------- WebSocket progress ----------

    @app.websocket("/api/jobs/{job_id}/ws")
    async def job_ws(websocket: WebSocket, job_id: str):
        """Stream job progress over WebSocket until done/failed."""
        await websocket.accept()
        try:
            while True:
                job = _get_job(job_id)
                if job is None:
                    await websocket.send_json({"error": "Job not found"})
                    break
                await websocket.send_json(
                    {
                        "status": job.status.value,
                        "stage": job.stage,
                        "progress": job.progress,
                    }
                )
                if job.status in (JobStatus.done, JobStatus.failed):
                    break
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            pass
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    # Mount static files if directory exists
    if static_dir is None:
        static_dir = str(Path(__file__).resolve().parent.parent.parent / "static")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


# Default app instance for `uvicorn pixelduet.web:app`
app = create_app()
