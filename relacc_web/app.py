from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .service import (
    create_job,
    export_result,
    get_job,
    parse_config,
    public_job,
    render_job_overlay,
    run_job,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="RelAcc Web Workbench", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/jobs")
async def create_evaluation_job(
    background_tasks: BackgroundTasks,
    reference_zip: UploadFile = File(...),
    candidate_zip: UploadFile = File(...),
    config: str = Form("{}"),
):
    parsed_config = parse_config(config)
    reference_bytes = await reference_zip.read()
    candidate_bytes = await candidate_zip.read()
    job = create_job(reference_bytes, candidate_bytes, parsed_config)
    if job["status"] == "queued":
        background_tasks.add_task(run_job, job["id"])
    return job


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return public_job(job)


@app.get("/api/jobs/{job_id}/results")
def read_results(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.get("result"):
        raise HTTPException(status_code=409, detail="Job has no completed results yet.")
    return job["result"]


@app.get("/api/jobs/{job_id}/exports/{fmt}")
def export(job_id: str, fmt: str):
    if fmt not in {"json", "csv"}:
        raise HTTPException(status_code=404, detail="Unsupported export format.")
    exported = export_result(job_id, fmt)
    if exported is None:
        raise HTTPException(status_code=404, detail="Export not found.")
    content, media_type = exported
    extension = "json" if fmt == "json" else "csv"
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="relacc-results.{extension}"'},
    )


@app.get("/api/jobs/{job_id}/overlay")
def overlay(job_id: str, key: str):
    svg = render_job_overlay(job_id, key)
    if svg is None:
        raise HTTPException(status_code=404, detail="Overlay not found.")
    return Response(svg, media_type="image/svg+xml")
