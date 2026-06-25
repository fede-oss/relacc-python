import json

from fastapi.testclient import TestClient

from relacc_web.app import app
from relacc_web import service
from relacc_web.service import JOBS, EvaluationConfig, create_group_job, run_job
from test.spec.relacc_web.test_service import _sample, _zip_bytes


def setup_function():
    JOBS.clear()


def teardown_function():
    JOBS.clear()


def test_jobs_endpoint_accepts_repeated_comparison_zips():
    client = TestClient(app)
    reference = _zip_bytes({"ref-a.csv": _sample(0), "ref-b.csv": _sample(1)})
    candidate_a = _zip_bytes({"cand-a.csv": _sample(2)})
    candidate_b = _zip_bytes({"cand-b.csv": _sample(3)})

    response = client.post(
        "/api/jobs",
        data={
            "comparison_names": json.dumps(["alpha", "beta"]),
            "config": json.dumps({"mode": "summary", "rate": 4}),
        },
        files=[
            ("reference_zip", ("reference.zip", reference, "application/zip")),
            ("comparison_zips", ("alpha.zip", candidate_a, "application/zip")),
            ("comparison_zips", ("beta.zip", candidate_b, "application/zip")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["validation"]["mode"]["groupCount"] == 2
    assert sorted(payload["validation"]["candidates"]) == ["alpha", "beta"]


def test_jobs_endpoint_rejects_invalid_alignment_before_creating_job():
    client = TestClient(app)
    jobs_before = set(JOBS)
    reference = _zip_bytes({"ref.csv": _sample(0)})
    candidate = _zip_bytes({"cand.csv": _sample(1)})

    response = client.post(
        "/api/jobs",
        data={"config": json.dumps({"alignment": 2})},
        files=[
            ("reference_zip", ("reference.zip", reference, "application/zip")),
            ("candidate_zip", ("candidate.zip", candidate, "application/zip")),
        ],
    )

    assert response.status_code == 400
    assert "alignment" in response.json()["detail"].lower()
    assert set(JOBS) == jobs_before


def test_jobs_endpoint_rejects_per_file_upload_limit(monkeypatch):
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 10)
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files=[
            ("reference_zip", ("reference.zip", b"x" * 11, "application/zip")),
            ("candidate_zip", ("candidate.zip", b"x", "application/zip")),
        ],
    )

    assert response.status_code == 413
    assert JOBS == {}


def test_jobs_endpoint_rejects_total_upload_limit(monkeypatch):
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(service, "MAX_TOTAL_UPLOAD_BYTES", 15)
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files=[
            ("reference_zip", ("reference.zip", b"x" * 8, "application/zip")),
            ("candidate_zip", ("candidate.zip", b"x" * 8, "application/zip")),
        ],
    )

    assert response.status_code == 413
    assert JOBS == {}


def test_jobs_endpoint_rejects_comparison_count_limit(monkeypatch):
    monkeypatch.setattr(service, "MAX_COMPARISON_ARCHIVES", 1)
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files=[
            ("reference_zip", ("reference.zip", b"x", "application/zip")),
            ("comparison_zips", ("a.zip", b"x", "application/zip")),
            ("comparison_zips", ("b.zip", b"x", "application/zip")),
        ],
    )

    assert response.status_code == 413
    assert JOBS == {}


def test_jobs_endpoint_rejects_active_job_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "MAX_ACTIVE_JOBS", 1)
    service.JOBS["active"] = {
        "id": "active",
        "status": "running",
        "updatedAt": 0,
        "workdir": str(tmp_path / "relacc-web-active"),
    }
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files=[
            ("reference_zip", ("reference.zip", _zip_bytes({"ref.csv": _sample()}), "application/zip")),
            ("candidate_zip", ("candidate.zip", _zip_bytes({"cand.csv": _sample(1)}), "application/zip")),
        ],
    )

    assert response.status_code == 429
    assert set(JOBS) == {"active"}


def test_jobs_endpoint_rejects_archive_expansion_limit(monkeypatch):
    monkeypatch.setattr(service, "MAX_MEMBER_UNCOMPRESSED_BYTES", 10)
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files=[
            ("reference_zip", ("reference.zip", _zip_bytes({"ref.csv": _sample()}), "application/zip")),
            ("candidate_zip", ("candidate.zip", _zip_bytes({"cand.csv": _sample(1)}), "application/zip")),
        ],
    )

    assert response.status_code == 413
    assert JOBS == {}


def test_read_job_returns_404_after_terminal_expiry(monkeypatch):
    monkeypatch.setattr(service, "JOB_TTL_SECONDS", 0)
    public = create_group_job(
        _zip_bytes({"ref.csv": _sample()}),
        [("candidate", _zip_bytes({"cand.csv": _sample(1)}))],
        EvaluationConfig(mode="summary"),
    )
    run_job(public["id"])
    client = TestClient(app)

    response = client.get(f"/api/jobs/{public['id']}")

    assert response.status_code == 404


def test_overlay_endpoint_returns_404_for_bad_group_overlay_key():
    public = create_group_job(
        _zip_bytes({"a.csv": _sample()}),
        [("alpha", _zip_bytes({"a.csv": _sample(1)})), ("beta", _zip_bytes({"a.csv": _sample(2)}))],
        EvaluationConfig(mode="direct", rate=4),
    )
    run_job(public["id"])
    client = TestClient(app)

    ok_response = client.get(f"/api/jobs/{public['id']}/overlay", params={"key": JOBS[public["id"]]["result"]["metadata"]["overlayKeys"][0]})
    bad_response = client.get(f"/api/jobs/{public['id']}/overlay", params={"key": "missing"})

    assert ok_response.status_code == 200
    assert ok_response.headers["content-type"].startswith("image/svg+xml")
    assert bad_response.status_code == 404
