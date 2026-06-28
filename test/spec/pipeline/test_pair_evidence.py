from relacc.pipeline import pair_evidence as PairEvidence


def _endpoint(key, path, point_id):
    return PairEvidence.PairEndpoint(key=key, path=path, points=[point_id])


def test_bidirectional_pair_evidence_averages_directions_and_builds_distribution_rows(
    monkeypatch,
):
    left = _endpoint("left.csv", "/tmp/left.csv", "left")
    right = _endpoint("nested/right.csv", "/tmp/right.csv", "right")
    values_by_points = {
        ("left", "right"): {"shapeError": 2.0, "dtwDistance": 6.0},
        ("right", "left"): {"shapeError": 4.0, "dtwDistance": 10.0},
    }
    calls = []

    def fake_compute_pair_metrics(reference_points, candidate_points, *args, **kwargs):
        calls.append((reference_points[0], candidate_points[0], kwargs["dtw_window"]))
        return values_by_points[(reference_points[0], candidate_points[0])]

    monkeypatch.setattr(
        PairEvidence,
        "compute_pair_metrics_from_points",
        fake_compute_pair_metrics,
    )

    evidence = PairEvidence.compute_bidirectional_pair_evidence(
        left,
        right,
        PairEvidence.PairMetricOptions(
            label="arrow",
            effective_rate=48,
            metric_names=("shapeError", "dtwDistance"),
            dtw_window=7,
        ),
    )

    assert evidence.values == {"shapeError": 3.0, "dtwDistance": 8.0}
    assert calls == [("left", "right", 7), ("right", "left", 7)]
    assert PairEvidence.joined_pair_key(left, right) == "left::nested/right"
    assert PairEvidence.joined_pair_path(left, right) == "/tmp/left.csv::/tmp/right.csv"

    rows = PairEvidence.distribution_within_reference_rows(
        evidence,
        {"schemaVersion": 1, "recordType": "rawMetricOutput"},
    )

    assert rows[0] == {
        "schemaVersion": 1,
        "recordType": "rawMetricOutput",
        "sampleKind": PairEvidence.WITHIN_REFERENCE_SAMPLE_KIND,
        "leftReferenceKey": "left.csv",
        "leftReferenceFile": "/tmp/left.csv",
        "rightReferenceKey": "nested/right.csv",
        "rightReferenceFile": "/tmp/right.csv",
        "metric": "shapeError",
        "value": 3.0,
        "forwardValue": 2.0,
        "backwardValue": 4.0,
    }


def test_directional_pair_evidence_builds_between_group_rows(monkeypatch):
    reference = _endpoint("ref.csv", "/tmp/ref.csv", "ref")
    comparison = _endpoint("cmp.csv", "/tmp/cmp.csv", "cmp")

    monkeypatch.setattr(
        PairEvidence,
        "compute_pair_metrics_from_points",
        lambda *args, **kwargs: {"shapeError": 1.25},
    )

    evidence = PairEvidence.compute_directional_pair_evidence(
        reference,
        comparison,
        PairEvidence.PairMetricOptions(
            label="arrow",
            effective_rate=24,
            metric_names=("shapeError",),
        ),
    )
    rows = PairEvidence.distribution_between_groups_rows(
        evidence,
        {"schemaVersion": 1, "recordType": "rawMetricOutput"},
    )

    assert evidence.direction == PairEvidence.REFERENCE_TO_CANDIDATE_DIRECTION
    assert rows == [
        {
            "schemaVersion": 1,
            "recordType": "rawMetricOutput",
            "sampleKind": PairEvidence.BETWEEN_GROUPS_SAMPLE_KIND,
            "referenceKey": "ref.csv",
            "referenceFile": "/tmp/ref.csv",
            "comparisonKey": "cmp.csv",
            "comparisonFile": "/tmp/cmp.csv",
            "metric": "shapeError",
            "value": 1.25,
        }
    ]
