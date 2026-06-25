from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .report_cache import (
    get_report_cache,
    list_report_caches,
    report_chart_payload,
    report_filters,
    report_overlay_svg,
    report_summary,
    report_table,
)
from .service import (
    create_group_job,
    create_job,
    export_result,
    get_job,
    parse_config,
    public_job,
    render_job_overlay,
    run_job,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
DIST_DIR = STATIC_DIR / "dist"
INDEX_FILE = DIST_DIR / "index.html"

app = FastAPI(title="RelAcc Web Workbench", version="0.1.0")
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/reports")
def reports():
    return {"reports": list_report_caches()}


@app.get("/api/reports/{report_id}/summary")
def cached_report_summary(report_id: str):
    report = get_report_cache(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report cache not found.")
    return report_summary(report)


@app.get("/api/reports/{report_id}/filters")
def cached_report_filters(report_id: str):
    report = get_report_cache(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report cache not found.")
    return report_filters(report)


@app.get("/api/reports/{report_id}/chart/{chart_kind}")
def cached_report_chart(
    report_id: str,
    chart_kind: str,
    source: str | None = None,
    dataset: str | None = None,
    variant: str | None = None,
    class_key: str | None = Query(default=None, alias="class"),
    metric: str | None = None,
    metrics: str | None = None,
    metric_family: str | None = None,
    distribution_metric: str | None = None,
):
    report = get_report_cache(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report cache not found.")
    payload = report_chart_payload(
        report,
        chart_kind,
        {
            "source": source,
            "dataset": dataset,
            "variant": variant,
            "classKey": class_key,
            "metric": metric,
            "metrics": metrics,
            "metricFamily": metric_family,
            "distributionMetric": distribution_metric,
        },
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Chart not found.")
    return payload


@app.get("/api/reports/{report_id}/table/{record_set}")
def cached_report_table(
    report_id: str,
    record_set: str,
    source: str | None = None,
    dataset: str | None = None,
    variant: str | None = None,
    class_key: str | None = Query(default=None, alias="class"),
    metric: str | None = None,
    limit: int = 200,
    offset: int = 0,
):
    report = get_report_cache(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report cache not found.")
    table = report_table(
        report,
        record_set,
        {
            "source": source,
            "dataset": dataset,
            "variant": variant,
            "classKey": class_key,
            "metric": metric,
        },
        limit=limit,
        offset=offset,
    )
    if table is None:
        raise HTTPException(status_code=404, detail="Table not found.")
    return table


@app.get("/api/reports/{report_id}/overlay")
def cached_report_overlay(
    report_id: str,
    source: str,
    dataset: str,
    class_key: str = Query(alias="class"),
    variant: str | None = None,
    comparison: str | None = None,
    sample_count: int = 18,
    summary: str | None = None,
    reference_color: str = "#0aa3a3",
    comparison_color: str = "#e53935",
    summary_color: str = "#111514",
    show_reference: bool = True,
    show_comparison: bool = True,
    show_summary: bool = True,
    summary_source: str = "reference",
):
    report = get_report_cache(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report cache not found.")
    svg = report_overlay_svg(
        report,
        source=source,
        dataset=dataset,
        variant=variant,
        class_key=class_key,
        comparison=comparison,
        sample_count=sample_count,
        summary=summary,
        reference_color=reference_color,
        comparison_color=comparison_color,
        summary_color=summary_color,
        show_reference=show_reference,
        show_comparison=show_comparison,
        show_summary=show_summary,
        summary_source=summary_source,
    )
    if svg is None:
        raise HTTPException(status_code=404, detail="Overlay not found.")
    return Response(svg, media_type="image/svg+xml")


@app.post("/api/jobs")
async def create_evaluation_job(
    background_tasks: BackgroundTasks,
    reference_zip: UploadFile = File(...),
    candidate_zip: UploadFile | None = File(default=None),
    comparison_zips: list[UploadFile] = File(default=[]),
    comparison_names: str = Form("[]"),
    config: str = Form("{}"),
):
    try:
        parsed_config = parse_config(config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    reference_bytes = await reference_zip.read()
    comparisons: list[tuple[str, bytes]] = []
    if candidate_zip is not None:
        comparisons.append(("candidate", await candidate_zip.read()))
    names = []
    try:
        import json

        names = json.loads(comparison_names or "[]")
    except Exception:
        names = []
    for index, upload in enumerate(comparison_zips, start=1):
        name = names[index - 1] if index - 1 < len(names) else f"comparison-{index}"
        comparisons.append((str(name), await upload.read()))
    if not comparisons:
        raise HTTPException(status_code=400, detail="At least one comparison ZIP is required.")
    job = create_group_job(reference_bytes, comparisons, parsed_config)
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
