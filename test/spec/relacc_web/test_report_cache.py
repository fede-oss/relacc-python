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
