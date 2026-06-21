import json

from fastapi.testclient import TestClient

from relacc_web.app import app
from test.spec.relacc_web.test_service import _sample, _zip_bytes


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
