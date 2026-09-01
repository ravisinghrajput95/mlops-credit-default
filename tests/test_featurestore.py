"""The feature store and its point-in-time guarantee.

The unfold is asserted lossless, because a transformation that quietly drops or
mis-dates an observation changes the modelling data and nothing downstream would
notice. Everything else here guards the same single rule: no feature may be built
from an event later than the row's as-of date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_default.featurestore.events import (
    MONTH_COLUMNS,
    OUTCOME_MONTH,
    entities,
    refold,
    statement_months,
    unfold,
)
from credit_default.featurestore.pit import (
    build_spine,
    latest_join,
    point_in_time_join,
)
from credit_default.featurestore.views import FEATURE_NAMES, aggregate
from tests.conftest import make_frame

APRIL = pd.Timestamp("2005-04-01")
JULY = pd.Timestamp("2005-07-01")
SEPTEMBER = pd.Timestamp("2005-09-01")


@pytest.fixture
def wide():
    return make_frame(rows=120, seed=3)


@pytest.fixture
def store(wide):
    return unfold(wide), entities(wide)


# ------------------------------------------------------------- unfold ------


def test_the_unfold_is_lossless(wide):
    """Round-trip or it did not happen: a lossy unfold would silently change the
    data every model downstream is trained on."""
    events, entity_frame = unfold(wide), entities(wide)
    restored = refold(events, entity_frame)
    for _, pay, bill, amount in MONTH_COLUMNS:
        for column in (pay, bill, amount):
            assert restored[column].tolist() == wide[column].tolist()


def test_every_customer_gets_one_row_per_month(wide):
    events = unfold(wide)
    assert len(events) == len(wide) * len(MONTH_COLUMNS)
    assert events.groupby("customer_id")["statement_month"].nunique().eq(6).all()


def test_september_maps_to_pay_0_not_pay_1(wide):
    """The trap the column names set. Deriving the month from the numeric suffix
    mis-dates a sixth of the observations, and nothing would flag it."""
    events = unfold(wide)
    september = events[events["statement_month"] == SEPTEMBER].set_index("customer_id")
    assert september["pay_status"].tolist() == wide["PAY_0"].tolist()
    assert september["bill_amt"].tolist() == wide["BILL_AMT1"].tolist()


def test_months_are_ordered_oldest_first():
    months = statement_months()
    assert months == sorted(months)
    assert months[-1] < OUTCOME_MONTH


# --------------------------------------------------------- leak guards -----


def test_no_feature_can_see_past_its_as_of_date(store):
    """The rule, stated directly."""
    events, entity_frame = store
    spine = build_spine(entity_frame["customer_id"], JULY)
    features = point_in_time_join(events, entity_frame, spine)
    # April through July inclusive.
    assert features["observed_months"].eq(4).all()


def test_the_window_grows_as_the_as_of_date_advances(store):
    events, entity_frame = store
    seen = [
        point_in_time_join(events, entity_frame, build_spine(entity_frame["customer_id"], month))[
            "observed_months"
        ].max()
        for month in statement_months()
    ]
    assert seen == [1, 2, 3, 4, 5, 6]


def test_the_naive_join_ignores_the_as_of_date_entirely(store):
    """It always returns the full history, which is exactly the bug."""
    events, entity_frame = store
    for month in statement_months():
        leaked = latest_join(events, entity_frame, build_spine(entity_frame["customer_id"], month))
        assert leaked["observed_months"].eq(6).all()


def test_the_two_joins_disagree_whenever_there_is_a_future_to_leak(store):
    """The canary. If these ever coincide before the last month, the guarantee has
    evaporated and every other test here would still pass."""
    events, entity_frame = store
    spine = build_spine(entity_frame["customer_id"], JULY)
    honest = point_in_time_join(events, entity_frame, spine).set_index("customer_id").sort_index()
    leaked = latest_join(events, entity_frame, spine).set_index("customer_id").sort_index()

    differing = ~np.isclose(
        honest[FEATURE_NAMES].to_numpy(dtype=float), leaked[FEATURE_NAMES].to_numpy(dtype=float)
    )
    assert differing.any()


def test_the_two_joins_agree_once_there_is_nothing_left_to_leak(store):
    """At the last observable month the correct answer and the careless one
    coincide -- which is why a leaky pipeline looks healthy until it is asked
    about the past."""
    events, entity_frame = store
    spine = build_spine(entity_frame["customer_id"], SEPTEMBER)
    honest = point_in_time_join(events, entity_frame, spine).set_index("customer_id").sort_index()
    leaked = latest_join(events, entity_frame, spine).set_index("customer_id").sort_index()

    np.testing.assert_allclose(
        honest[FEATURE_NAMES].to_numpy(dtype=float),
        leaked[FEATURE_NAMES].to_numpy(dtype=float),
    )


def test_an_as_of_at_or_after_the_outcome_is_refused(store):
    """Features dated at or after the outcome are the outcome."""
    _, entity_frame = store
    with pytest.raises(ValueError, match="outcome month"):
        build_spine(entity_frame["customer_id"], OUTCOME_MONTH)
    with pytest.raises(ValueError, match="outcome month"):
        build_spine(entity_frame["customer_id"], "2005-11-01")


def test_a_spine_without_an_as_of_date_is_refused(store):
    events, entity_frame = store
    spine = pd.DataFrame({"customer_id": entity_frame["customer_id"]})
    with pytest.raises(ValueError, match="as-of"):
        point_in_time_join(events, entity_frame, spine)


def test_a_later_event_cannot_change_an_earlier_rows_features(store):
    """Appending future history must leave a past as-of untouched. This is the
    property a real store breaks first, when a backfill silently rewrites what
    yesterday's model would have seen."""
    events, entity_frame = store
    spine = build_spine(entity_frame["customer_id"], APRIL)
    before = point_in_time_join(events, entity_frame, spine).set_index("customer_id").sort_index()

    # A wildly out-of-range observation, one month after the last real one.
    intruder = events[events["statement_month"] == SEPTEMBER].copy()
    intruder["statement_month"] = pd.Timestamp("2005-09-30")
    intruder["bill_amt"] = 10_000_000
    after = (
        point_in_time_join(
            events=pd.concat([events, intruder], ignore_index=True),
            entity_frame=entity_frame,
            spine=spine,
        )
        .set_index("customer_id")
        .sort_index()
    )
    np.testing.assert_allclose(
        before[FEATURE_NAMES].to_numpy(dtype=float), after[FEATURE_NAMES].to_numpy(dtype=float)
    )


# ---------------------------------------------------------- aggregates -----


def test_aggregates_are_computed_over_the_window_they_are_given():
    events = pd.DataFrame(
        {
            "customer_id": [1, 1, 1],
            "statement_month": pd.to_datetime(["2005-04-01", "2005-05-01", "2005-06-01"]),
            "pay_status": [0, 2, 1],
            "bill_amt": [100, 200, 400],
            "pay_amt": [50, 0, 100],
        }
    )
    row = aggregate(events).iloc[0]
    assert row["observed_months"] == 3
    assert row["pay_status_latest"] == 1  # June, not the maximum
    assert row["pay_status_max"] == 2
    assert row["months_delinquent"] == 2
    assert row["bill_latest"] == 400
    assert row["bill_trend"] == 300  # 400 - 100


def test_latest_means_most_recent_not_last_row_encountered():
    """Ordering is enforced inside aggregate, so a shuffled event table cannot
    silently change what 'latest' means."""
    events = pd.DataFrame(
        {
            "customer_id": [1, 1, 1],
            "statement_month": pd.to_datetime(["2005-06-01", "2005-04-01", "2005-05-01"]),
            "pay_status": [1, 0, 2],
            "bill_amt": [400, 100, 200],
            "pay_amt": [100, 50, 0],
        }
    )
    assert aggregate(events).iloc[0]["bill_latest"] == 400


def test_an_empty_window_produces_no_rows_rather_than_nan():
    empty = pd.DataFrame(
        columns=["customer_id", "statement_month", "pay_status", "bill_amt", "pay_amt"]
    )
    assert aggregate(empty).empty


def test_utilisation_survives_a_zero_credit_limit(wide):
    """Division guarded rather than producing inf, which XGBoost accepts and then
    behaves unpredictably on."""
    wide = wide.copy()
    wide.loc[wide.index[0], "LIMIT_BAL"] = 0
    events, entity_frame = unfold(wide), entities(wide)
    features = point_in_time_join(
        events, entity_frame, build_spine(entity_frame["customer_id"], SEPTEMBER)
    )
    assert np.isfinite(features["utilisation_latest"]).all()


def test_every_declared_feature_is_actually_produced(store):
    events, entity_frame = store
    features = point_in_time_join(
        events, entity_frame, build_spine(entity_frame["customer_id"], JULY)
    )
    assert set(FEATURE_NAMES) <= set(features.columns)
    assert not features[FEATURE_NAMES].isna().any().any()
