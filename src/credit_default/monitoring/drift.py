"""Data drift monitoring.

Compares the reference cohort the model was trained on against a "current"
window, and fails when too large a share of monitored columns has moved.

Why drift is monitored rather than accuracy: in credit default the ground-truth
label -- whether the customer actually defaulted -- is only known months after the
prediction is served. Waiting for labels means finding out about a broken model a
quarter late. Input drift and prediction-score drift are the signals available
*now*, so they are what the alerting is built on.

The current window can come from the held-out cohort (for demos) or from what the
API actually served (Postgres locally, Parquet on GCS in the cloud).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from credit_default.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

DRIFTED_COUNT_METRIC = "DriftedColumnsCount"
VALUE_DRIFT_METRIC = "evidently:metric_v2:ValueDrift"


def _as_dataset(frame: pd.DataFrame) -> Dataset:
    """Declare column roles explicitly so PAY_* are tested as categories.

    Left to inference, the integer-coded repayment-status columns would be
    treated as numeric and compared with a distance measure that is meaningless
    for ordinal codes.
    """
    definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )
    return Dataset.from_pandas(frame[FEATURES], data_definition=definition)


def load_current(settings: Settings, source: str) -> pd.DataFrame:
    """Load the window to compare against reference."""
    if source == "cohort":
        return pd.read_parquet(settings.current_parquet)

    if source == "postgres":
        import psycopg

        with psycopg.connect(settings.postgres_dsn) as conn:
            rows = conn.execute(
                "SELECT features FROM predictions ORDER BY predicted_at DESC LIMIT 10000"
            ).fetchall()
        if not rows:
            raise ValueError("No served predictions found in Postgres yet.")
        return pd.DataFrame([r[0] for r in rows])

    if source == "gcs":
        prefix = settings.gcs_prediction_prefix.rstrip("/")
        frame = pd.read_parquet(f"{prefix}/")
        # The sink flattens the feature dict into "features.<NAME>" columns.
        renamed = {c: c.split("features.", 1)[-1] for c in frame.columns if "features." in c}
        return frame.rename(columns=renamed)[FEATURES]

    raise ValueError(f"Unknown current-data source: {source}")


def summarise(report_dict: dict[str, Any]) -> dict[str, Any]:
    """Reduce the Evidently payload to the few numbers alerting needs."""
    drifted_share = 0.0
    drifted_count = 0
    columns: dict[str, dict[str, float | bool]] = {}

    for metric in report_dict.get("metrics", []):
        name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")

        if name.startswith(DRIFTED_COUNT_METRIC) and isinstance(value, dict):
            drifted_count = int(value.get("count", 0))
            drifted_share = float(value.get("share", 0.0))
        elif config.get("type") == VALUE_DRIFT_METRIC:
            threshold = float(config.get("threshold", 0.0))
            score = float(value) if value is not None else 0.0
            columns[str(config.get("column"))] = {
                "score": round(score, 5),
                "method": config.get("method", ""),
                "threshold": threshold,
                "drifted": score > threshold,
            }

    return {
        "drifted_columns": drifted_count,
        "monitored_columns": len(columns),
        "drifted_share": round(drifted_share, 4),
        "columns": columns,
    }


def run(source: str = "cohort", settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    settings.ensure_dirs()

    reference = pd.read_parquet(settings.reference_parquet)
    current = load_current(settings, source)
    logger.info("Comparing %d reference rows against %d current rows", len(reference), len(current))

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        current_data=_as_dataset(current),
        reference_data=_as_dataset(reference),
    )

    html_path = settings.reports_dir / "drift_report.html"
    result.save_html(str(html_path))

    summary = summarise(result.dict())
    summary["source"] = source
    summary["reference_rows"] = len(reference)
    summary["current_rows"] = len(current)
    summary["threshold"] = settings.max_drifted_share
    summary["passed"] = summary["drifted_share"] <= settings.max_drifted_share
    summary["html_report"] = str(html_path)

    (settings.reports_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data drift check.")
    parser.add_argument(
        "--source",
        choices=["cohort", "postgres", "gcs"],
        default="cohort",
        help="Where the current window comes from.",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit non-zero when drift exceeds the threshold (used by scheduled CI).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    summary = run(args.source)

    drifted = [name for name, info in summary["columns"].items() if info["drifted"]]
    logger.info(
        "Drift: %d/%d columns (share %.3f, threshold %.3f) -> %s",
        summary["drifted_columns"],
        summary["monitored_columns"],
        summary["drifted_share"],
        summary["threshold"],
        "PASS" if summary["passed"] else "FAIL",
    )
    if drifted:
        logger.info("Drifted columns: %s", ", ".join(sorted(drifted)))
    logger.info("Report written to %s", Path(summary["html_report"]).resolve())

    if args.fail_on_drift and not summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
