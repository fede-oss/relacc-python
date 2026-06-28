from __future__ import annotations

import json
from typing import Dict

from relacc.distribution_metrics import DISTRIBUTION_METRIC_NAMES
from relacc.metrics import METRIC_NAMES


STATISTICAL_MODE = "descriptive-pair-distances"
INDEPENDENT_UNIT = "gesture-file"
PAIR_VALUES_INDEPENDENT = False
STATISTICS_SCHEMA_VERSION = 2
REMOVED_INFERENTIAL_FIELDS = (
    "meanCi95Low",
    "meanCi95High",
    "normalityPValue",
    "ksPValue",
)

PAIRWISE_COLUMNS = (
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "pairKey",
    "candidateFile",
    "referenceInput",
    "mode",
    "referenceCount",
    "candidateCount",
    "rate",
    "requestedRate",
    "alignment",
    "alignmentName",
    "summary",
    "popular",
    "dtwWindow",
    "exactDtw",
    *METRIC_NAMES,
)

BASELINE_COLUMNS = (
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "sampleKey",
    "sampleFile",
    "referenceInput",
    "mode",
    "referenceCount",
    "rate",
    "requestedRate",
    "alignment",
    "alignmentName",
    "summary",
    "popular",
    "dtwWindow",
    "exactDtw",
    *METRIC_NAMES,
)

STATS_COLUMNS = (
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "summary",
    "metric",
    "n",
    "finiteN",
    "mean",
    "mdn",
    "sd",
    "min",
    "max",
)

DISTRIBUTION_COLUMNS = (
    "statisticalMode",
    "independentUnit",
    "pairValuesIndependent",
    "statisticsSchemaVersion",
    "removedInferentialFields",
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "summary",
    "metric",
    "withinReferenceN",
    "withinReferenceFiniteN",
    "withinReferenceMean",
    "withinReferenceMdn",
    "withinReferenceSd",
    "withinReferenceVariance",
    "withinReferenceMin",
    "withinReferenceMax",
    "withinReferenceQ05",
    "withinReferenceQ25",
    "withinReferenceQ50",
    "withinReferenceQ75",
    "withinReferenceQ95",
    "withinReferenceSkewness",
    "withinReferenceKurtosis",
    "withinComparisonN",
    "withinComparisonFiniteN",
    "withinComparisonMean",
    "withinComparisonMdn",
    "withinComparisonSd",
    "withinComparisonVariance",
    "withinComparisonMin",
    "withinComparisonMax",
    "withinComparisonQ05",
    "withinComparisonQ25",
    "withinComparisonQ50",
    "withinComparisonQ75",
    "withinComparisonQ95",
    "withinComparisonSkewness",
    "withinComparisonKurtosis",
    "betweenGroupsN",
    "betweenGroupsFiniteN",
    "betweenGroupsMean",
    "betweenGroupsMdn",
    "betweenGroupsSd",
    "betweenGroupsVariance",
    "betweenGroupsMin",
    "betweenGroupsMax",
    "betweenGroupsQ05",
    "betweenGroupsQ25",
    "betweenGroupsQ50",
    "betweenGroupsQ75",
    "betweenGroupsQ95",
    "betweenGroupsSkewness",
    "betweenGroupsKurtosis",
    *DISTRIBUTION_METRIC_NAMES,
    "normalizedWassersteinDistance",
    "betweenGroupsMeanDelta",
    "betweenGroupsMdnDelta",
    "betweenGroupsSdDelta",
    "withinComparisonToReferenceMeanDelta",
    "withinComparisonToReferenceMeanRatio",
    "withinComparisonToReferenceMdnDelta",
    "withinComparisonToReferenceMdnRatio",
    "withinComparisonToReferenceSdDelta",
    "withinComparisonToReferenceSdRatio",
)

COMBINED_OUTPUT_DIRNAME = "combined"
COMBINED_PAIRWISE_FILENAME = "pairwise.csv"
COMBINED_STATS_FILENAME = "stats.csv"
COMBINED_BASELINE_FILENAME = "baseline.csv"
COMBINED_BASELINE_STATS_FILENAME = "baseline_stats.csv"
COMBINED_WITHIN_REFERENCE_FILENAME = "within_reference.csv"
COMBINED_WITHIN_REFERENCE_STATS_FILENAME = "within_reference_stats.csv"
COMBINED_WITHIN_COMPARISON_FILENAME = "within_comparison.csv"
COMBINED_WITHIN_COMPARISON_STATS_FILENAME = "within_comparison_stats.csv"
COMBINED_BETWEEN_GROUPS_FILENAME = "between_groups.csv"
COMBINED_BETWEEN_GROUPS_STATS_FILENAME = "between_groups_stats.csv"
COMBINED_DISTRIBUTION_FILENAME = "distribution.csv"
COMBINED_SUMMARY_DISTRIBUTION_FILENAME = "summary_distribution.csv"
COMBINED_AGGREGATE_SUMMARIES_FILENAME = "aggregate_summaries.csv"
COMBINED_RAW_METRICS_FILENAME = "raw_metrics.jsonl"
COMBINED_RAW_DISTRIBUTIONS_FILENAME = "raw_distributions.jsonl"
COMBINED_REPORT_FILENAME = "report.json"

AGGREGATE_SUMMARY_COLUMNS = (
    "recordSet",
    "scope",
    "source",
    "dataset",
    "variant",
    "summary",
    "metric",
    "n",
    "finiteN",
    "mean",
    "mdn",
    "sd",
    "min",
    "max",
)

DISTRIBUTION_DERIVED_VALUE_COLUMNS = (
    "normalizedWassersteinDistance",
    "betweenGroupsMeanDelta",
    "betweenGroupsMdnDelta",
    "betweenGroupsSdDelta",
    "withinComparisonToReferenceMeanDelta",
    "withinComparisonToReferenceMeanRatio",
    "withinComparisonToReferenceMdnDelta",
    "withinComparisonToReferenceMdnRatio",
    "withinComparisonToReferenceSdDelta",
    "withinComparisonToReferenceSdRatio",
)

DISTRIBUTION_OUTPUT_VALUE_COLUMNS = (
    *DISTRIBUTION_METRIC_NAMES,
    *DISTRIBUTION_DERIVED_VALUE_COLUMNS,
)


def statistical_contract_fields() -> Dict[str, object]:
    return {
        "statisticalMode": STATISTICAL_MODE,
        "independentUnit": INDEPENDENT_UNIT,
        "pairValuesIndependent": PAIR_VALUES_INDEPENDENT,
        "statisticsSchemaVersion": STATISTICS_SCHEMA_VERSION,
        "removedInferentialFields": list(REMOVED_INFERENTIAL_FIELDS),
    }


def statistical_contract_csv_fields() -> Dict[str, object]:
    return {
        "statisticalMode": STATISTICAL_MODE,
        "independentUnit": INDEPENDENT_UNIT,
        "pairValuesIndependent": PAIR_VALUES_INDEPENDENT,
        "statisticsSchemaVersion": STATISTICS_SCHEMA_VERSION,
        "removedInferentialFields": json.dumps(
            list(REMOVED_INFERENTIAL_FIELDS),
            separators=(",", ":"),
        ),
    }
