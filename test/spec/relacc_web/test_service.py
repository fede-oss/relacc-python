import io
import zipfile

from relacc_web.service import EvaluationConfig, create_job, get_job, run_job


def _zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buf.getvalue()


def _sample(offset=0):
    return "\n".join(
        [
            "stroke_id x y time is_writing",
            f"0 {10+offset} 20 0 1",
            f"0 {12+offset} 22 10 1",
            f"1 {14+offset} 24 20 1",
        ]
    )


def test_create_and_run_summary_job():
    reference = _zip_bytes({"ref-a.csv": _sample(0), "ref-b.csv": _sample(1)})
    candidate = _zip_bytes({"cand-a.csv": _sample(2)})

    public = create_job(reference, candidate, EvaluationConfig(mode="summary", rate=4))
    assert public["status"] == "queued"

    run_job(public["id"])
    job = get_job(public["id"])
    assert job["status"] == "completed"
    assert job["result"]["metadata"]["comparisonMode"] == "summary"
    assert job["result"]["metadata"]["pairCount"] == 1


def test_create_job_reports_bad_zip():
    public = create_job(b"not a zip", b"not a zip", EvaluationConfig())
    assert public["status"] == "failed"
    assert public["validation"]["issues"][0]["severity"] == "error"
