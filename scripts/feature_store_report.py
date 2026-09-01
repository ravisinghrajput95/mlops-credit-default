#!/usr/bin/env python
"""Measure what a point-in-time-incorrect feature join actually costs.

Three numbers per as-of date, because the two obvious ones tell less than half
the story:

``leaked (reported)``
    Train and evaluate on features aggregated over every month the store holds,
    ignoring the as-of date. This is the number a leaky pipeline puts in a deck.

``leaked model, honest features``
    The same leaked model, scored on the features it will actually receive in
    production -- only the months that existed at decision time. This is what the
    deployment gets, and it is the number nobody computes.

``honest (reported = production)``
    Train and evaluate point-in-time correctly. Lower than the first, and the
    only one of the three that means what it says.

The middle column is the point. Leakage is usually described as inflating your
estimate, which sounds like a reporting problem. It is worse than that: a model
trained on months it will not have at serving time learns to depend on them, so
it is *actively worse in production* than the honest model it beat on paper. The
error is not in the metric, it is in the weights.

    python scripts/feature_store_report.py
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from credit_default.config import TARGET, get_settings
from credit_default.featurestore.events import entities, statement_months, unfold
from credit_default.featurestore.pit import build_spine, latest_join, point_in_time_join
from credit_default.featurestore.views import FEATURE_NAMES

logger = logging.getLogger(__name__)

RULE = "=" * 92


def fit(train: pd.DataFrame, seed: int) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_NAMES], train[TARGET])
    return model


def score(model: XGBClassifier, test: pd.DataFrame) -> tuple[float, float]:
    probabilities = model.predict_proba(test[FEATURE_NAMES])[:, 1]
    return (
        float(average_precision_score(test[TARGET], probabilities)),
        float(roc_auc_score(test[TARGET], probabilities)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    settings = get_settings()
    settings.ensure_dirs()

    raw = pd.read_parquet(settings.raw_parquet)
    events = unfold(raw)
    entity_frame = entities(raw, target=TARGET)

    train_ids, test_ids = train_test_split(
        entity_frame["customer_id"],
        test_size=settings.test_fraction,
        random_state=settings.random_seed,
        stratify=entity_frame[TARGET],
    )
    labels = entity_frame.set_index("customer_id")[TARGET]

    print(
        f"\n{len(raw):,} customers unfolded into {len(events):,} monthly events "
        f"({len(statement_months())} months, April-September 2005)."
    )
    print(f"Outcome is default in October 2005. Train {len(train_ids):,} / test {len(test_ids):,}.")

    print("\n" + RULE)
    print("WHAT A POINT-IN-TIME-INCORRECT JOIN BUYS, AND WHAT IT ACTUALLY COSTS")
    print(RULE)
    print(
        f"{'as of':<12}{'months':>9}{'corrupted':>12}"
        f"{'leaked':>12}{'leaked model,':>18}{'honest':>10}{'production':>13}"
    )
    print(
        f"{'':<12}{'visible':>9}{'values':>12}{'(reported)':>12}{'honest features':>18}"
        f"{'(real)':>10}{'penalty':>13}"
    )

    rows = []
    for as_of in statement_months():
        spine = build_spine(entity_frame["customer_id"], as_of)
        honest = point_in_time_join(events, entity_frame, spine)
        leaked = latest_join(events, entity_frame, spine)

        honest = honest.assign(**{TARGET: honest["customer_id"].map(labels)})
        leaked = leaked.assign(**{TARGET: leaked["customer_id"].map(labels)})

        # How much of the feature matrix the two joins actually disagree on.
        aligned_h = honest.set_index("customer_id")[FEATURE_NAMES].sort_index()
        aligned_l = leaked.set_index("customer_id")[FEATURE_NAMES].sort_index()
        corrupted = float(
            (~np.isclose(aligned_h.to_numpy(dtype=float), aligned_l.to_numpy(dtype=float))).mean()
        )

        h_train = honest[honest["customer_id"].isin(train_ids)]
        h_test = honest[honest["customer_id"].isin(test_ids)]
        l_train = leaked[leaked["customer_id"].isin(train_ids)]
        l_test = leaked[leaked["customer_id"].isin(test_ids)]

        leaked_model = fit(l_train, settings.random_seed)
        honest_model = fit(h_train, settings.random_seed)

        leaked_reported, _ = score(leaked_model, l_test)
        # The same leaked model, handed the features production can actually
        # supply. Nothing about the model changed; only what it is fed.
        leaked_in_production, _ = score(leaked_model, h_test)
        honest_reported, honest_auc = score(honest_model, h_test)

        months = int(honest["observed_months"].max())
        rows.append(
            {
                "as_of": str(as_of.date()),
                "months_visible": months,
                "corrupted_feature_share": round(corrupted, 4),
                "leaked_reported_pr_auc": round(leaked_reported, 4),
                "leaked_in_production_pr_auc": round(leaked_in_production, 4),
                "honest_pr_auc": round(honest_reported, 4),
                "honest_roc_auc": round(honest_auc, 4),
                "overstatement": round(leaked_reported - honest_reported, 4),
                "production_penalty": round(leaked_in_production - honest_reported, 4),
            }
        )
        print(
            f"{as_of.date()!s:<12}{months:>9}{corrupted:>11.0%}"
            f"{leaked_reported:>12.4f}{leaked_in_production:>18.4f}"
            f"{honest_reported:>10.4f}{leaked_in_production - honest_reported:>+13.4f}"
        )

    payload = {
        "customers": len(raw),
        "events": len(events),
        "months": [str(m.date()) for m in statement_months()],
        "features": FEATURE_NAMES,
        "as_of": rows,
    }

    early = rows[0]
    print(
        f"\n  At {early['as_of']} the two joins disagree on "
        f"{early['corrupted_feature_share']:.0%} of the feature matrix. The leaked model "
        f"reports\n  {early['leaked_reported_pr_auc']:.4f} against an honest "
        f"{early['honest_pr_auc']:.4f} -- an overstatement of "
        f"{early['overstatement']:+.4f}."
    )
    penalties = [r["production_penalty"] for r in rows if r["months_visible"] < 6]
    if penalties:
        print(
            f"\n  Handed the features production can actually supply, that same leaked model "
            f"scores\n  {min(penalties):+.4f} to {max(penalties):+.4f} against the honest one. "
            "Leakage does not merely\n  inflate the estimate -- it ships a worse model, because "
            "the weights were fitted to\n  months that will not be there."
        )
    last = rows[-1]
    print(
        f"\n  At {last['as_of']} the joins agree ({last['corrupted_feature_share']:.0%} corrupted) "
        f"and the numbers converge.\n  Nothing is left to leak once the as-of date reaches the "
        "last observation, which is why\n  a leaky pipeline looks perfectly healthy right up "
        "until it is asked about the past."
    )

    destination = args.output or (settings.reports_dir / "feature_store.json")
    with open(destination, "w") as handle:  # noqa: PTH123
        json.dump(payload, handle, indent=2)
    print(f"\nWritten to {destination}")


if __name__ == "__main__":
    main()
