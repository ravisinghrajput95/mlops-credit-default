#!/usr/bin/env python
"""Replay a held-out cohort as served traffic and let its outcomes arrive late.

Two things happen here, and only one of them is invented.

**Real.** The held-out ``current`` cohort is scored by the actual champion model
at its actual cost-derived threshold. Which applicants get declined -- and
therefore never generate an outcome at all -- is a genuine consequence of the
model's own policy applied to real data, not a scenario chosen to make a point.
The labels are the real UCI ground truth.

**Simulated.** *When* each decision was made and *when* each outcome came back.
The dataset is a static snapshot with no time dimension whatsoever, so there is
nothing to measure here and every timestamp is generated from the documented
assumptions in ``labels.arrival``.

Decisions are anchored to a fixed synthetic epoch rather than to "now" so that
the report is reproducible: a table in the README should not quietly change
because a day passed.

    python scripts/simulate_label_arrival.py
    python scripts/simulate_label_arrival.py --holdout 0.05 --span-days 365
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import numpy as np
import pandas as pd

from credit_default.api.model import load_model
from credit_default.config import TARGET, get_settings
from credit_default.labels.arrival import (
    SYNTHETIC_WARNING,
    ArrivalModel,
    assign_decision_times,
    emit_outcomes,
)
from credit_default.labels.store import build_outcome_store

logger = logging.getLogger(__name__)

# Fixed synthetic epoch. See the module docstring: reproducibility beats realism
# for a number that ends up in documentation.
DEFAULT_ORIGIN = "2024-01-01"
DEFAULT_SPAN_DAYS = 180


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN, help="Date of the first decision.")
    parser.add_argument("--span-days", type=int, default=DEFAULT_SPAN_DAYS)
    parser.add_argument(
        "--holdout",
        type=float,
        default=None,
        help="Share of would-be declines approved anyway to keep that range observable.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    settings.ensure_dirs()

    holdout = args.holdout if args.holdout is not None else settings.label_holdout_fraction
    seed = args.seed if args.seed is not None else settings.random_seed

    print("\n" + "!" * 78)
    print("  " + SYNTHETIC_WARNING.replace(". ", ".\n  "))
    print("!" * 78 + "\n")

    cohort = pd.read_parquet(settings.current_parquet)
    handle = load_model(settings)
    probabilities, decisions = handle.predict(cohort)

    rng_origin = dt.datetime.fromisoformat(args.origin).replace(tzinfo=dt.UTC)
    rng = np.random.default_rng(seed)

    served = pd.DataFrame(
        {
            # Deterministic keys, so a rerun joins to the outcomes already recorded
            # rather than orphaning them.
            "application_id": [f"APP-{i:06d}" for i in range(len(cohort))],
            "predicted_at": assign_decision_times(len(cohort), rng_origin, args.span_days, rng),
            "model_version": handle.version,
            "probability": probabilities,
            "prediction": decisions,
        }
    )
    served.to_parquet(settings.served_predictions_parquet, index=False)

    model = ArrivalModel(
        performance_window_days=settings.label_performance_window_days,
        reporting_lag_days=settings.label_reporting_lag_days,
        reporting_lag_default_days=settings.label_reporting_lag_default_days,
    )
    outcomes, detail = emit_outcomes(served, cohort[TARGET], model, holdout, seed)

    store = build_outcome_store(settings)
    written = store.record(outcomes)

    # The truth frame is the demo's answer key: it holds the outcome for every
    # applicant including the ones the policy declined, which is precisely the
    # information a production system does not have.
    detail[
        ["application_id", "probability", "prediction", "label", "observable", "holdout"]
    ].to_parquet(settings.processed_dir / "label_truth.parquet", index=False)

    declined = int((served["prediction"] == 1).sum())
    print(f"{'decisions replayed':<38}{len(served):>10,}")
    print(
        f"{'declined by policy (threshold ' + format(handle.threshold, '.2f') + ')':<38}"
        f"{declined:>10,}  ({declined / len(served):.1%})"
    )
    print(
        f"{'holdout approvals (rate ' + format(holdout, '.1%') + ')':<38}"
        f"{int(detail['holdout'].sum()):>10,}"
    )
    print(
        f"{'outcomes that will ever exist':<38}{len(outcomes):>10,}"
        f"  ({len(outcomes) / len(served):.1%} of decisions)"
    )
    print(f"{'outcomes newly recorded':<38}{written:>10,}")
    print(f"\nServed predictions -> {settings.served_predictions_parquet}")
    print(f"Outcome store      -> {settings.outcomes_parquet}")
    print("\nNow run: make label-report")


if __name__ == "__main__":
    main()
