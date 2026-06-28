import csv
import json

from relacc_web import report_cache


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_report(tmp_path):
    root = tmp_path / "report-output-eval-test"
    combined = root / "combined"
    combined.mkdir(parents=True)
    manifest = {
        "summary": "medoid",
        "rate": 24,
        "alignment": 1,
        "metricNames": ["shapeError", "dtwDistance"],
        "distributionMetricNames": ["wassersteinDistance"],
        "runs": [
            {
                "source": "DHG",
                "dataset": "1dollar",
                "variant": "root",
                "classes": [{"classKey": "arrow", "outputDir": "report-output-eval-test/DHG/1dollar/classes/arrow"}],
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "run.json").write_text(json.dumps({"createdUtc": "today", "effectiveConfig": {"summary": "medoid"}}), encoding="utf-8")
    distribution_row = {
        "source": "DHG",
        "dataset": "1dollar",
        "variant": "root",
        "classKey": "arrow",
        "summary": "medoid",
        "metric": "shapeError",
        "withinReferenceMean": "1",
        "withinComparisonMean": "1.4",
        "betweenGroupsMean": "2",
        "withinReferenceQ25": "0.8",
        "withinReferenceQ75": "1.2",
        "betweenGroupsQ25": "1.5",
        "betweenGroupsQ75": "2.5",
        "withinComparisonToReferenceMeanRatio": "1.4",
        "wassersteinDistance": "0.5",
        "normalizedWassersteinDistance": "0.25",
    }
    _write_csv(combined / "distribution.csv", [distribution_row])
    _write_csv(combined / "summary_distribution.csv", [distribution_row])
    stat_row = {
        "source": "DHG",
        "dataset": "1dollar",
        "variant": "root",
        "classKey": "arrow",
        "summary": "medoid",
        "metric": "shapeError",
        "n": "2",
        "finiteN": "2",
        "mean": "1",
        "mdn": "1",
        "sd": "0",
        "min": "1",
        "max": "1",
    }
    _write_csv(combined / "baseline_stats.csv", [stat_row])
    _write_csv(combined / "stats.csv", [{**stat_row, "mdn": "1.5"}])
    for name in report_cache.COMBINED_FILES:
        path = combined / report_cache.COMBINED_FILES[name]
        if not path.exists():
            _write_csv(path, [distribution_row if "distribution" in name else stat_row])
    return root


def _load_report(root):
    return report_cache.ReportCache(
        id="test",
        root=root,
        combined=root / "combined",
        manifest=json.loads((root / "manifest.json").read_text(encoding="utf-8")),
        run=json.loads((root / "run.json").read_text(encoding="utf-8")),
    )


def _set_output_dir(root, output_dir):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["runs"][0]["classes"][0]["outputDir"] = output_dir
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _class_dir(report):
    return report_cache._class_dir(report, "DHG", "1dollar", "root", "arrow")


def test_report_cache_discovers_and_builds_chart_payloads(tmp_path, monkeypatch):
    root = _sample_report(tmp_path)
    monkeypatch.setattr(report_cache, "PROJECT_REPORT_ROOT", tmp_path)
    monkeypatch.setenv("RELACC_REPORT_ROOTS", str(tmp_path))

    reports = report_cache.list_report_caches()
    assert reports[0]["name"] == root.name

    report = report_cache.get_report_cache(reports[0]["id"])
    filters = {"metric": "shapeError", "metricFamily": "core"}
    ranking = report_cache.report_chart_payload(report, "ranking", filters)
    scatter = report_cache.report_chart_payload(report, "scatter", filters)

    assert ranking["rows"][0]["source"] == "DHG"
    assert scatter["points"][0]["generatedMedian"] == 1.5


def test_manifest_output_dir_rejects_absolute_path_outside_report(tmp_path):
    root = _sample_report(tmp_path)
    outside_dir = tmp_path / "outside-absolute"
    outside_dir.mkdir()
    _set_output_dir(root, str(outside_dir))

    assert _class_dir(_load_report(root)) is None


def test_manifest_output_dir_rejects_relative_traversal(tmp_path):
    root = _sample_report(tmp_path)
    (tmp_path / "outside").mkdir()
    _set_output_dir(root, "../outside")

    assert _class_dir(_load_report(root)) is None


def test_manifest_output_dir_allows_absolute_path_inside_report(tmp_path):
    root = _sample_report(tmp_path)
    class_dir = root / "DHG" / "1dollar" / "classes" / "arrow"
    class_dir.mkdir(parents=True)
    _set_output_dir(root, str(class_dir))

    assert _class_dir(_load_report(root)) == class_dir.resolve()


def test_overlay_omits_absolute_sample_files_outside_allowed_roots(tmp_path, monkeypatch):
    root = _sample_report(tmp_path)
    class_dir = root / "DHG" / "1dollar" / "classes" / "arrow"
    class_dir.mkdir(parents=True)
    inside_file = class_dir / "inside.csv"
    outside_dir = tmp_path.parent / f"{tmp_path.name}-overlay-outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "outside.csv"
    inside_file.write_text("x,y\n0,0\n", encoding="utf-8")
    outside_file.write_text("x,y\n1,1\n", encoding="utf-8")
    _write_csv(class_dir / "baseline.csv", [{"sampleFile": str(outside_file)}, {"sampleFile": str(inside_file)}])
    _write_csv(class_dir / "pairwise.csv", [{"candidateFile": str(outside_file)}])
    monkeypatch.delenv("RELACC_OVERLAY_FILE_ROOTS", raising=False)
    captured = {}

    def capture_overlay(groups, **kwargs):
        captured["groups"] = groups
        return "<svg></svg>"

    monkeypatch.setattr(report_cache, "render_overlay_svg", capture_overlay)

    svg = report_cache.report_overlay_svg(
        _load_report(root),
        "DHG",
        "1dollar",
        "root",
        "arrow",
        None,
        10,
        None,
        "#000",
        "#111",
    )

    assert svg == "<svg></svg>"
    assert captured["groups"][0].files == [str(inside_file.resolve())]
    assert captured["groups"][1].files == []


def test_overlay_allows_absolute_sample_files_under_env_roots(tmp_path, monkeypatch):
    root = _sample_report(tmp_path)
    class_dir = root / "DHG" / "1dollar" / "classes" / "arrow"
    class_dir.mkdir(parents=True)
    outside_dir = tmp_path.parent / f"{tmp_path.name}-overlay-allowed"
    outside_dir.mkdir()
    outside_file = outside_dir / "outside.csv"
    outside_file.write_text("x,y\n1,1\n", encoding="utf-8")
    _write_csv(class_dir / "baseline.csv", [{"sampleFile": str(outside_file)}])
    _write_csv(class_dir / "pairwise.csv", [{"candidateFile": str(outside_file)}])
    monkeypatch.setenv("RELACC_OVERLAY_FILE_ROOTS", str(outside_dir))
    captured = {}

    def capture_overlay(groups, **kwargs):
        captured["groups"] = groups
        return "<svg></svg>"

    monkeypatch.setattr(report_cache, "render_overlay_svg", capture_overlay)

    svg = report_cache.report_overlay_svg(
        _load_report(root),
        "DHG",
        "1dollar",
        "root",
        "arrow",
        None,
        10,
        None,
        "#000",
        "#111",
    )

    assert svg == "<svg></svg>"
    assert captured["groups"][0].files == [str(outside_file.resolve())]
    assert captured["groups"][1].files == [str(outside_file.resolve())]
