import io
import os
import time
import zipfile
from pathlib import Path

import pytest

from relacc.pipeline import pairwise as Pairwise
from relacc_web import service
from relacc_web.service import (
    EvaluationConfig,
    create_group_job,
    create_job,
    get_job,
    prune_terminal_jobs,
    render_job_overlay,
    run_job,
)


@pytest.fixture(autouse=True)
def clear_jobs():
    service.JOBS.clear()
    yield
    service.JOBS.clear()


def _zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buf.getvalue()


def _zip_bytes_with_infos(infos):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for info, text in infos:
            archive.writestr(info, text)
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


def test_direct_job_validation_uses_pipeline_layout_matching():
    reference = _zip_bytes(
        {"projected3Dsignatures/realTO/78_14.csv": _sample(0)}
    )
    candidate = _zip_bytes({"projected3Dsignatures/78_14.csv": _sample(2)})

    public = create_job(reference, candidate, EvaluationConfig(mode="direct"))

    assert public["status"] == "queued"
    assert public["validation"]["mode"]["matchedPairCount"] == 1
    assert public["validation"]["mode"]["missingInCandidate"] == []
    assert public["validation"]["mode"]["missingInReference"] == []

    run_job(public["id"])
    job = get_job(public["id"])
    assert job["status"] == "completed"
    assert job["result"]["metadata"]["pairCount"] == 1


def test_create_group_job_accepts_multiple_comparisons():
    reference = _zip_bytes({"ref-a.csv": _sample(0), "ref-b.csv": _sample(1)})
    candidate_a = _zip_bytes({"cand-a.csv": _sample(2)})
    candidate_b = _zip_bytes({"cand-b.csv": _sample(3)})

    public = create_group_job(
        reference,
        [("alpha", candidate_a), ("beta", candidate_b)],
        EvaluationConfig(mode="summary", rate=4),
    )

    assert public["status"] == "queued"
    assert public["validation"]["mode"]["groupCount"] == 2
    assert sorted(public["validation"]["candidates"]) == ["alpha", "beta"]


@pytest.mark.parametrize(
    "limit_name,limit_value,files,should_error",
    [
        ("MAX_ZIP_MEMBERS", 2, {"a.csv": _sample(), "b.csv": _sample()}, False),
        ("MAX_ZIP_MEMBERS", 1, {"a.csv": _sample(), "b.csv": _sample()}, True),
        ("MAX_CSV_FILES", 2, {"a.csv": _sample(), "b.csv": _sample()}, False),
        ("MAX_CSV_FILES", 1, {"a.csv": _sample(), "b.csv": _sample()}, True),
        ("MAX_MEMBER_UNCOMPRESSED_BYTES", len(_sample()), {"a.csv": _sample()}, False),
        ("MAX_MEMBER_UNCOMPRESSED_BYTES", len(_sample()) - 1, {"a.csv": _sample()}, True),
        ("MAX_ARCHIVE_UNCOMPRESSED_BYTES", len(_sample()) * 2, {"a.csv": _sample(), "b.csv": _sample()}, False),
        ("MAX_ARCHIVE_UNCOMPRESSED_BYTES", len(_sample()) * 2 - 1, {"a.csv": _sample(), "b.csv": _sample()}, True),
    ],
)
def test_extract_zip_declared_boundaries(monkeypatch, tmp_path, limit_name, limit_value, files, should_error):
    monkeypatch.setattr(service, limit_name, limit_value)

    issues, extracted = service.extract_zip(_zip_bytes(files), tmp_path, "candidate")

    assert any(issue["severity"] == "error" for issue in issues) is should_error
    if should_error:
        assert extracted == []
        assert list(tmp_path.rglob("*.csv")) == []
    else:
        assert len(extracted) == len(files)


def test_extract_zip_rejects_compression_ratio(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "MAX_COMPRESSION_RATIO", 1)

    issues, extracted = service.extract_zip(_zip_bytes({"a.csv": _sample() * 30}), tmp_path, "candidate")

    assert extracted == []
    assert any("compression ratio" in issue["message"].lower() for issue in issues)


def test_extract_zip_rejects_encrypted_and_symlink_like_members(monkeypatch, tmp_path):
    symlink = zipfile.ZipInfo("link.csv")
    symlink.external_attr = (0o120777 << 16)
    data = _zip_bytes_with_infos([(symlink, _sample())])

    issues, extracted = service.extract_zip(data, tmp_path, "candidate")

    assert extracted == []
    assert any("symlink" in issue["message"].lower() for issue in issues)

    real_infolist = zipfile.ZipFile.infolist

    def encrypted_infolist(self):
        infos = real_infolist(self)
        infos[0].flag_bits |= 0x1
        return infos

    monkeypatch.setattr(zipfile.ZipFile, "infolist", encrypted_infolist)
    issues, extracted = service.extract_zip(_zip_bytes({"encrypted.csv": _sample()}), tmp_path, "candidate")

    assert extracted == []
    assert any("encrypted" in issue["message"].lower() for issue in issues)


def test_extract_zip_streamed_overrun_rolls_back(monkeypatch, tmp_path):
    data = _zip_bytes({"a.csv": _sample()})
    real_open = zipfile.ZipFile.open

    def lying_open(self, name, mode="r", pwd=None, *, force_zip64=False):
        source = real_open(self, name, mode=mode, pwd=pwd, force_zip64=force_zip64)
        if mode == "r":
            return io.BytesIO(source.read() + b"overflow")
        return source

    monkeypatch.setattr(zipfile.ZipFile, "open", lying_open)

    issues, extracted = service.extract_zip(data, tmp_path, "candidate")

    assert extracted == []
    assert any("exceeded" in issue["message"].lower() for issue in issues)
    assert list(tmp_path.rglob("*.csv")) == []


def test_failed_validation_workdir_is_cleaned():
    public = create_job(b"not a zip", b"not a zip", EvaluationConfig())
    assert public["status"] == "failed"
    job = service.JOBS[public["id"]]
    assert not Path(job["workdir"]).exists()


def test_pruning_terminal_jobs_preserves_active_and_safely_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "TEMP_ROOT", tmp_path)
    monkeypatch.setattr(service, "JOB_TTL_SECONDS", 10)
    monkeypatch.setattr(service, "MAX_RETAINED_TERMINAL_JOBS", 1)
    now = time.time()
    old_dir = tmp_path / "relacc-web-old-abcd"
    old_dir.mkdir()
    recent_dir = tmp_path / "relacc-web-recent-abcd"
    recent_dir.mkdir()
    active_dir = tmp_path / "relacc-web-active-abcd"
    active_dir.mkdir()
    unsafe_dir = tmp_path / "not-relacc-web"
    unsafe_dir.mkdir()
    service.JOBS.update(
        {
            "old": {"id": "old", "status": "completed", "updatedAt": now - 20, "workdir": str(old_dir)},
            "recent": {"id": "recent", "status": "completed", "updatedAt": now, "workdir": str(recent_dir)},
            "active": {"id": "active", "status": "running", "updatedAt": now - 20, "workdir": str(active_dir)},
            "unsafe": {"id": "unsafe", "status": "failed", "updatedAt": now - 20, "workdir": str(unsafe_dir)},
        }
    )

    prune_terminal_jobs(now=now)

    assert "old" not in service.JOBS
    assert "recent" in service.JOBS
    assert "active" in service.JOBS
    assert "unsafe" not in service.JOBS
    assert not old_dir.exists()
    assert recent_dir.exists()
    assert active_dir.exists()
    assert unsafe_dir.exists()


def test_startup_cleanup_only_removes_stale_relacc_children(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "TEMP_ROOT", tmp_path)
    monkeypatch.setattr(service, "JOB_TTL_SECONDS", 10)
    stale = tmp_path / "relacc-web-stale-abcd"
    fresh = tmp_path / "relacc-web-fresh-abcd"
    unsafe = tmp_path / "other"
    for path in (stale, fresh, unsafe):
        path.mkdir()
    old_time = time.time() - 20
    os.utime(stale, (old_time, old_time))

    service.cleanup_orphan_workdirs(now=time.time())

    assert not stale.exists()
    assert fresh.exists()
    assert unsafe.exists()


def _write_tree(root: Path, files):
    for name, rows in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows), encoding="utf-8")


@pytest.mark.parametrize(
    "reference_files,candidate_files",
    [
        ({"a.csv": None}, {"a.csv": None}),
        ({"reference-only-layout/a.csv": None}, {"candidate-only-layout/a.csv": None}),
        ({"alpha/realTO/same.csv": None, "beta/realTO/same.csv": None}, {"alpha/same.csv": None, "beta/same.csv": None}),
        ({"one/a.csv": None, "two/a.csv": None}, {"candidate/a.csv": None}),
        ({"a.csv": None, "only-ref.csv": None}, {"a.csv": None, "only-cand.csv": None}),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_direct_validation_matches_pairwise_discovery(tmp_path, reference_files, candidate_files, strict):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_tree(reference_dir, {name: Pairwise._sample_rows() if hasattr(Pairwise, "_sample_rows") else _sample().splitlines() for name in reference_files})
    _write_tree(candidate_dir, {name: _sample(1).splitlines() for name in candidate_files})

    try:
        pairs, missing_candidate, missing_reference = Pairwise.discover_pairs(
            str(reference_dir),
            str(candidate_dir),
            strict=False,
        )
        expected_pair_count = len(pairs)
    except ValueError:
        missing_candidate = []
        missing_reference = []
        expected_pair_count = 0

    validation = service.validate_inputs(reference_dir, candidate_dir, EvaluationConfig(mode="direct", strict=strict))

    assert validation["mode"]["matchedPairCount"] == expected_pair_count
    assert validation["mode"]["missingInCandidate"] == missing_candidate[:50]
    assert validation["mode"]["missingInReference"] == missing_reference[:50]
    expected_ok = expected_pair_count > 0 and (not strict or (not missing_candidate and not missing_reference))
    assert validation["ok"] is expected_ok


def test_group_job_validates_each_direct_comparison_before_queueing():
    reference = _zip_bytes({"layout/a.csv": _sample(), "layout/b.csv": _sample(1)})
    valid = _zip_bytes({"other/a.csv": _sample(2), "other/b.csv": _sample(3)})
    invalid = _zip_bytes({"missing.csv": _sample(4)})

    public = create_group_job(reference, [("valid", valid), ("invalid", invalid)], EvaluationConfig(mode="direct"))

    assert public["status"] == "failed"
    assert "invalid" in public["validation"]["candidates"]
    assert any(issue["scope"] == "invalid:matching" for issue in public["validation"]["issues"])


def test_group_pairwise_overlay_keys_are_unique_and_render():
    reference = _zip_bytes({"a.csv": _sample()})
    candidate_a = _zip_bytes({"a.csv": _sample(2)})
    candidate_b = _zip_bytes({"a.csv": _sample(3)})
    public = create_group_job(
        reference,
        [("alpha::one", candidate_a), ("beta/two", candidate_b)],
        EvaluationConfig(mode="direct", rate=4),
    )
    run_job(public["id"])
    job = get_job(public["id"])
    keys = job["result"]["metadata"]["overlayKeys"]

    assert len(keys) == 2
    assert len(set(keys)) == 2
    assert all("overlayKey" in row for row in job["result"]["pairs"])
    assert all("<svg" in render_job_overlay(public["id"], key) for key in keys)
    assert render_job_overlay(public["id"], "malformed") is None
    assert render_job_overlay(public["id"], service.encode_group_overlay_key("missing", "a")) is None


def test_group_distribution_overlay_keys_are_unique_and_render():
    reference = _zip_bytes({"class/a.csv": _sample(), "class/b.csv": _sample(1)})
    candidate_a = _zip_bytes({"class/c.csv": _sample(2)})
    candidate_b = _zip_bytes({"class/d.csv": _sample(3)})
    public = create_group_job(
        reference,
        [("alpha", candidate_a), ("beta", candidate_b)],
        EvaluationConfig(mode="distribution", group_by="parent-dir", rate=4),
    )
    run_job(public["id"])
    job = get_job(public["id"])
    keys = job["result"]["metadata"]["overlayKeys"]

    assert len(keys) == 2
    assert len(set(keys)) == 2
    assert all("<svg" in render_job_overlay(public["id"], key) for key in keys)
