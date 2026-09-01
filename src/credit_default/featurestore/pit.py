"""The as-of join, and the one everybody writes instead.

A training row is a question asked at a moment: *given what was known about this
customer on this date, will they default?* The features must therefore be
assembled from the months on or before that date and no others. This is the whole
of point-in-time correctness, and it is easy to state and easy to lose.

``latest_join`` is how it gets lost. It aggregates every event the store holds,
which is what ``GROUP BY customer_id`` does when nobody has thought about time,
and it is what a feature table queried today returns for a label from last
quarter. The result is not subtly optimistic. It is a model scored on months that
had not happened yet when the decision was made, and it cannot be reproduced in
production at any price -- at serving time those rows simply do not exist.

Both are exported. The measurement in ``scripts/feature_store_report.py`` trains
on each and reports the gap, because "leakage inflates your metrics" is worth
very little as an assertion and a great deal as a number.
"""

from __future__ import annotations

import logging

import pandas as pd

from credit_default.featurestore.events import OUTCOME_MONTH
from credit_default.featurestore.views import aggregate, attach_entity_features

logger = logging.getLogger(__name__)

SPINE_COLUMNS = ["customer_id", "as_of_month"]


def build_spine(customer_ids: pd.Series, as_of: pd.Timestamp | str) -> pd.DataFrame:
    """Ask the same question of every customer on the same date.

    A real spine carries a different as-of per row -- each application is scored
    when it arrives. A single shared date is used here because the demo compares
    what the model knows at different points in the same history, and varying two
    things at once would make the comparison unreadable.
    """
    as_of = pd.Timestamp(as_of)
    if as_of >= OUTCOME_MONTH:
        raise ValueError(
            f"as_of {as_of.date()} is not before the outcome month "
            f"{OUTCOME_MONTH.date()}. Features dated at or after the outcome are "
            "the outcome, and no join can make that legitimate."
        )
    return pd.DataFrame({"customer_id": list(customer_ids), "as_of_month": as_of})


def point_in_time_join(
    events: pd.DataFrame, entity_frame: pd.DataFrame, spine: pd.DataFrame
) -> pd.DataFrame:
    """Features as they stood on each row's as-of date.

    Grouped by as-of rather than evaluated per row: the spine holds a handful of
    distinct dates, so one filter per date replaces one per customer. The
    guarantee is unchanged -- no row ever sees an event after its own as-of.
    """
    missing = [column for column in SPINE_COLUMNS if column not in spine.columns]
    if missing:
        raise ValueError(f"Spine is missing {missing}; a row with no as-of date cannot be joined.")

    events = events.copy()
    events["statement_month"] = pd.to_datetime(events["statement_month"])

    frames = []
    for as_of, rows in spine.groupby("as_of_month", sort=True):
        visible = events[
            events["statement_month"] <= pd.Timestamp(as_of)  # type: ignore[arg-type]
        ]
        visible = visible[visible["customer_id"].isin(rows["customer_id"])]
        features = aggregate(visible)
        features["as_of_month"] = as_of
        frames.append(features)

    if not frames:
        return pd.DataFrame(columns=[*SPINE_COLUMNS, *aggregate(events.head(0)).columns])

    joined = pd.concat(frames, ignore_index=True)
    result = attach_entity_features(joined, entity_frame)
    logger.info(
        "Point-in-time join: %d rows across %d as-of date(s), %d months visible at the latest",
        len(result),
        spine["as_of_month"].nunique(),
        int(result["observed_months"].max()) if len(result) else 0,
    )
    return result


def latest_join(
    events: pd.DataFrame, entity_frame: pd.DataFrame, spine: pd.DataFrame
) -> pd.DataFrame:
    """The WRONG join, kept so the cost of getting it wrong can be measured.

    Every event the store holds, regardless of the as-of date on the row being
    built. This is not a strawman: it is what ``GROUP BY customer_id`` returns,
    what a feature table queried today gives you for a label from last quarter,
    and what almost every first version of a feature pipeline does.

    ``tests/test_featurestore.py`` asserts it disagrees with the correct join
    wherever the as-of date precedes the last event, so that if the two ever
    coincide the suite fails rather than the guarantee silently evaporating.
    """
    visible = events[events["customer_id"].isin(spine["customer_id"])]
    features = aggregate(visible)
    features = features.merge(spine[SPINE_COLUMNS], on="customer_id", how="right")
    return attach_entity_features(features, entity_frame)
