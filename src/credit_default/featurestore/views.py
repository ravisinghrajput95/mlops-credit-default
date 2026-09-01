"""Feature definitions.

Every feature here is an aggregate over a *window of months*, which is precisely
why point-in-time correctness is not a theoretical concern in this project. A
single-month attribute is hard to leak: it either belongs to the as-of month or
it does not. An aggregate leaks silently, because widening the window by one
month changes the value without changing the column name, the dtype, or anything
else a schema check would notice.

``observed_months`` is deliberately a feature rather than an internal detail. A
customer scored in May has two months of history and one scored in September has
six, and a model that cannot see how much history it is being given will read a
short window as a quiet customer rather than a new one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Everything an aggregate is computed from. Anything absent here cannot leak,
# because it never reaches a feature.
SOURCE_COLUMNS = ["pay_status", "bill_amt", "pay_amt"]

EVENT_FEATURES = [
    "observed_months",
    "pay_status_latest",
    "pay_status_max",
    "months_delinquent",
    "bill_latest",
    "bill_mean",
    "bill_trend",
    "pay_amt_latest",
    "pay_amt_mean",
    "payment_ratio_mean",
]

ENTITY_FEATURES = ["LIMIT_BAL", "utilisation_latest"]

FEATURE_NAMES = EVENT_FEATURES + ENTITY_FEATURES


def aggregate(window: pd.DataFrame) -> pd.DataFrame:
    """Collapse a window of events to one row per customer.

    ``window`` must already be restricted to the months a caller is allowed to
    see; this function does no filtering of its own and is not where correctness
    is enforced. Keeping the two apart is deliberate -- an aggregate that also
    decided its own visibility would make the rule impossible to test in
    isolation, and the rule is the part that matters.
    """
    if window.empty:
        return pd.DataFrame(columns=["customer_id", *EVENT_FEATURES])

    ordered = window.sort_values(["customer_id", "statement_month"])
    grouped = ordered.groupby("customer_id", sort=True)

    # Payment ratio is computed per month and then averaged, not as a ratio of
    # sums: a customer who pays a large bill once and nothing for five months is
    # not the same risk as one who pays steadily, and the ratio of sums cannot
    # tell them apart.
    ratio = ordered["pay_amt"] / ordered["bill_amt"].clip(lower=1)
    ordered = ordered.assign(_ratio=ratio.replace([np.inf, -np.inf], np.nan))

    features = pd.DataFrame(
        {
            "observed_months": grouped["statement_month"].count(),
            "pay_status_latest": grouped["pay_status"].last(),
            "pay_status_max": grouped["pay_status"].max(),
            "months_delinquent": grouped["pay_status"].apply(lambda s: int((s > 0).sum())),
            "bill_latest": grouped["bill_amt"].last(),
            "bill_mean": grouped["bill_amt"].mean(),
            "bill_trend": grouped["bill_amt"].last() - grouped["bill_amt"].first(),
            "pay_amt_latest": grouped["pay_amt"].last(),
            "pay_amt_mean": grouped["pay_amt"].mean(),
            "payment_ratio_mean": ordered.groupby("customer_id", sort=True)["_ratio"].mean(),
        }
    )
    return features.reset_index()


def attach_entity_features(features: pd.DataFrame, entity_frame: pd.DataFrame) -> pd.DataFrame:
    """Join the non-time-varying attributes and the ratios that need them.

    ``LIMIT_BAL`` is treated as static because this dataset gives it no time
    dimension. That is an assumption inherited from the source, not a modelling
    choice: a real credit line moves, and a store that recorded it as static
    would silently backdate today's limit onto last quarter's decision.
    """
    merged = features.merge(entity_frame, on="customer_id", how="left")
    merged["utilisation_latest"] = merged["bill_latest"] / merged["LIMIT_BAL"].clip(lower=1)
    return merged
