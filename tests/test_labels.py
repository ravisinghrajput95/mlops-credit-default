"""The delayed-label pipeline.

The tests that matter here are the negative ones. A join that returns rows is
easy; a join that refuses to return rows it should not have known about is the
entire product, and it fails silently when it breaks -- the metrics simply get
better. So the leakage guarantees are asserted directly, and the deliberately
wrong implementation is asserted to *disagree* with the correct one, which means
the suite fails if the guarantee is ever quietly removed.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from credit_default.config import Settings
from credit_default.labels.arrival import ArrivalModel, emit_outcomes, select_for_observation
from credit_default.labels.backfill import (
    join_labels,
    join_labels_ignoring_time,
    load_served_predictions,
)
from credit_default.labels.performance import (
    bootstrap_std_error,
    compare,
    estimate,
    selection_probability,
)
from credit_default.labels.store import (
    NullOutcomeStore,
    Outcome,
    ParquetOutcomeStore,
    build_outcome_store,
)

EPOCH = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
WINDOW = 30


def days(n: float) -> dt.timedelta:
    return dt.timedelta(days=n)


def make_predictions(rows: int = 6, declined_from: int = 4) -> pd.DataFrame:
    """Decisions one day apart; the last few are declined."""
    return pd.DataFrame(
        {
            "application_id": [f"APP-{i}" for i in range(rows)],
            "predicted_at": [EPOCH + days(i) for i in range(rows)],
            "probability": np.linspace(0.05, 0.95, rows),
            "prediction": [0] * declined_from + [1] * (rows - declined_from),
        }
    )


def make_outcomes(predictions: pd.DataFrame, reporting_lag: float = 10) -> pd.DataFrame:
    matures = pd.to_datetime(predictions["predicted_at"], utc=True) + pd.Timedelta(
        float(WINDOW), unit="D"
    )
    return pd.DataFrame(
        {
            "application_id": predictions["application_id"],
            "outcome_at": matures,
            "observed_at": matures + pd.Timedelta(float(reporting_lag), unit="D"),
            "label": [0, 1, 0, 1, 0, 1][: len(predictions)],
        }
    )


# --------------------------------------------------------------- store ------


def test_parquet_store_round_trips(tmp_path):
    store = ParquetOutcomeStore(tmp_path / "outcomes.parquet")
    store.record([Outcome("APP-0", EPOCH, EPOCH + days(10), 1)])
    frame = store.read()
    assert list(frame["application_id"]) == ["APP-0"]
    assert frame["label"].tolist() == [1]


def test_recording_the_same_outcome_twice_is_a_no_op(tmp_path):
    """A replayed feed must not restate history."""
    store = ParquetOutcomeStore(tmp_path / "outcomes.parquet")
    outcome = Outcome("APP-0", EPOCH, EPOCH + days(10), 1)
    assert store.record([outcome]) == 1
    assert store.record([outcome]) == 0
    assert len(store.read()) == 1


def test_a_restated_label_is_ignored_rather_than_applied(tmp_path):
    """An outcome is a historical fact. Changing one silently would corrupt every
    evaluation already computed against it."""
    store = ParquetOutcomeStore(tmp_path / "outcomes.parquet")
    store.record([Outcome("APP-0", EPOCH, EPOCH + days(10), 1)])
    store.record([Outcome("APP-0", EPOCH, EPOCH + days(10), 0)])
    assert store.read()["label"].tolist() == [1]


def test_reads_are_filtered_on_observation_not_on_the_outcome_date(tmp_path):
    """The bitemporal guarantee: the store answers "what did we know", not
    "what was true". An outcome that had happened but had not been reported is
    invisible to a read dated before it landed."""
    store = ParquetOutcomeStore(tmp_path / "outcomes.parquet")
    store.record([Outcome("APP-0", outcome_at=EPOCH, observed_at=EPOCH + days(60), label=1)])

    # The outcome_at date has long passed...
    assert store.read(as_of=EPOCH + days(30)).empty
    # ...but the report only lands on day 60.
    assert len(store.read(as_of=EPOCH + days(61))) == 1


def test_empty_store_returns_a_typed_frame(tmp_path):
    frame = ParquetOutcomeStore(tmp_path / "nothing.parquet").read()
    assert frame.empty
    assert list(frame.columns) == ["application_id", "outcome_at", "observed_at", "label"]


def test_unreachable_postgres_degrades_to_a_null_store():
    """Reporting must not be able to fail a training run."""
    settings = Settings(
        label_store="postgres", postgres_dsn="postgresql://nobody@127.0.0.1:1/nothing"
    )
    assert isinstance(build_outcome_store(settings), NullOutcomeStore)


def test_null_store_reads_and_writes_silently():
    store = NullOutcomeStore()
    assert store.record([Outcome("APP-0", EPOCH, EPOCH, 1)]) == 0
    assert store.read().empty


# ------------------------------------------------------ leakage guards ------


def test_a_prediction_whose_window_has_not_closed_is_excluded():
    """Gate 1: before the window closes the outcome is not merely unknown, it is
    undefined. Nothing about that applicant is scoreable yet."""
    predictions = make_predictions()
    outcomes = make_outcomes(predictions, reporting_lag=0)

    join = join_labels(predictions, outcomes, EPOCH + days(WINDOW - 1), WINDOW)
    assert join.matured == 0
    assert join.frame.empty


def test_an_outcome_that_exists_but_has_not_been_reported_is_excluded():
    """Gate 2, the subtle one. The label is sitting in the store with a real
    value; it simply had not arrived on the date being simulated. A join that
    picks it up produces a backtest production can never reproduce."""
    predictions = make_predictions(rows=1, declined_from=1)
    outcomes = make_outcomes(predictions, reporting_lag=45)

    # Window closed on day 30; the file lands on day 75.
    mid = EPOCH + days(50)
    join = join_labels(predictions, outcomes, mid, WINDOW)
    assert join.matured == 1
    assert join.observed == 0
    assert join.coverage == 0.0

    later = join_labels(predictions, outcomes, EPOCH + days(80), WINDOW)
    assert later.observed == 1


def test_the_join_filters_even_when_handed_unfiltered_outcomes():
    """The store normally filters on read. The join must not depend on that:
    one careless caller passing a full frame would otherwise leak silently."""
    predictions = make_predictions(rows=1, declined_from=1)
    everything = make_outcomes(predictions, reporting_lag=45)
    assert join_labels(predictions, everything, EPOCH + days(50), WINDOW).observed == 0


def test_the_naive_join_disagrees_with_the_correct_one():
    """If these ever coincide, the guarantee has evaporated and every test above
    would still pass. This is the canary for that."""
    predictions = make_predictions()
    outcomes = make_outcomes(predictions, reporting_lag=45)
    as_of = EPOCH + days(50)

    correct = join_labels(predictions, outcomes, as_of, WINDOW)
    naive = join_labels_ignoring_time(predictions, outcomes)

    assert len(naive) > correct.observed
    assert naive["observed_at"].max() > pd.Timestamp(as_of)


def test_predictions_without_a_key_are_rejected_with_a_useful_message():
    """A prediction nobody can name can never be joined to its outcome, and
    failing loudly at the join beats returning an empty frame."""
    predictions = make_predictions().drop(columns=["application_id"])
    with pytest.raises(ValueError, match="application_id"):
        join_labels(predictions, make_outcomes(make_predictions()), EPOCH + days(90), WINDOW)


def test_censored_and_pending_are_counted_separately():
    """A declined applicant's outcome is not late, it does not exist. Rolling the
    two together would make coverage look recoverable when it is not."""
    predictions = make_predictions(rows=6, declined_from=4)
    approved_only = make_outcomes(predictions).iloc[:2]

    join = join_labels(predictions, approved_only, EPOCH + days(90), WINDOW)
    assert join.matured == 6
    assert join.observed == 2
    assert join.censored == 2  # APP-4 and APP-5, declined
    assert join.pending == 2  # APP-2 and APP-3, approved but unreported
    assert join.censored + join.pending + join.observed == join.matured


def test_coverage_and_maturity_are_safe_when_nothing_qualifies():
    join = join_labels(make_predictions(), make_outcomes(make_predictions()), EPOCH, WINDOW)
    assert join.coverage == 0.0
    assert join.maturity == 0.0


def test_missing_served_predictions_name_the_command_that_creates_them(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="labels-replay"):
        load_served_predictions(settings, "parquet")


# ------------------------------------------------------------ arrival -------


def test_defaults_are_reported_more_slowly_than_non_defaults():
    """The documented asymmetry, asserted rather than assumed: it is the reason
    an immature cohort understates risk instead of just being smaller."""
    model = ArrivalModel(
        performance_window_days=WINDOW, reporting_lag_days=45, reporting_lag_default_days=75
    )
    rng = np.random.default_rng(0)
    labels = np.array([0] * 2000 + [1] * 2000)
    lags = model.reporting_lag(labels, rng)
    assert lags[labels == 1].mean() > lags[labels == 0].mean()
    assert (lags > 0).all()  # nothing is reported before it happens


def test_declined_applicants_are_observable_only_through_the_holdout():
    predictions = pd.Series([0] * 100 + [1] * 100)
    rng = np.random.default_rng(0)

    none_held_out = select_for_observation(predictions, 1e-12, rng)
    assert none_held_out[:100].all()
    assert not none_held_out[100:].any()

    all_held_out = select_for_observation(predictions, 1.0, rng)
    assert all_held_out.all()


def test_no_outcome_arrives_before_it_happens():
    predictions = make_predictions(rows=200, declined_from=200)
    model = ArrivalModel(
        performance_window_days=WINDOW, reporting_lag_days=45, reporting_lag_default_days=75
    )
    truth = pd.Series(np.random.default_rng(0).integers(0, 2, 200))
    outcomes, detail = emit_outcomes(predictions, truth, model, holdout_fraction=1.0, seed=0)

    assert len(outcomes) == 200
    assert all(o.observed_at > o.outcome_at for o in outcomes)
    assert (detail["matures_at"] > detail["predicted_at"]).all()


def test_censored_applicants_get_no_outcome_row():
    predictions = make_predictions(rows=100, declined_from=50)
    model = ArrivalModel(
        performance_window_days=WINDOW, reporting_lag_days=45, reporting_lag_default_days=75
    )
    truth = pd.Series(np.zeros(100, dtype=int))
    outcomes, detail = emit_outcomes(predictions, truth, model, holdout_fraction=1e-12, seed=0)

    assert len(outcomes) == 50
    assert detail.loc[~detail["observable"], "observed_at"].isna().all()


# -------------------------------------------------------- performance -------


def test_selection_probability_follows_the_served_decision():
    """Derivable from the prediction record alone, which is what makes the
    correction usable in production rather than only in this demo."""
    probabilities = selection_probability(pd.Series([0, 1, 0, 1]), 0.02)
    assert probabilities.tolist() == [1.0, 0.02, 1.0, 0.02]


def test_a_zero_holdout_is_rejected_because_nothing_can_recover_it():
    with pytest.raises(ValueError, match="holdout"):
        selection_probability(pd.Series([0, 1]), 0.0)


def test_reweighting_recovers_the_population_default_rate():
    """The property the whole correction rests on. A censored sample understates
    the true default rate because the model declined the risky applicants;
    Horvitz-Thompson weights put them back."""
    rng = np.random.default_rng(0)
    rows = 20_000
    probability = rng.uniform(0, 1, rows)
    label = (rng.uniform(0, 1, rows) < probability).astype(int)
    decision = (probability >= 0.3).astype(int)

    holdout = 0.1
    observed = rng.uniform(0, 1, rows) < np.where(decision == 1, holdout, 1.0)
    sample = pd.DataFrame(
        {
            "label": label[observed],
            "probability": probability[observed],
            "prediction": decision[observed],
        }
    )

    naive = sample.loc[sample["prediction"] == 0, "label"].mean()
    weights = 1.0 / selection_probability(sample["prediction"], holdout)
    corrected = np.average(sample["label"], weights=weights)

    assert naive < label.mean() - 0.05  # the censored view is badly biased low
    assert corrected == pytest.approx(label.mean(), abs=0.02)


def test_effective_rows_falls_when_weights_are_unequal():
    """Reweighting buys back the population and spends precision doing it. An
    IPW number reported without this is more confident than it has earned."""
    labels = np.array([0, 1] * 50)
    probabilities = np.linspace(0.01, 0.99, 100)
    equal = estimate(labels, probabilities, np.ones(100))
    lopsided = estimate(labels, probabilities, np.where(labels == 1, 50.0, 1.0))

    assert equal.effective_rows == pytest.approx(100.0)
    assert lopsided.effective_rows < equal.effective_rows * 0.6


def test_metrics_are_nan_rather_than_an_exception_on_a_single_class():
    """Early cohorts genuinely contain no defaults yet. A reporting job must not
    crash on a situation that is expected."""
    result = estimate(np.zeros(10, dtype=int), np.linspace(0, 1, 10), name="early")
    assert np.isnan(result.pr_auc)
    assert np.isnan(result.roc_auc)
    assert not np.isnan(result.brier)


def test_compare_reports_observed_and_corrected_side_by_side():
    joined = pd.DataFrame(
        {
            "label": [0, 1, 0, 1, 1, 0],
            "probability": [0.05, 0.1, 0.15, 0.6, 0.8, 0.9],
            "prediction": [0, 0, 0, 1, 1, 1],
        }
    )
    estimates = compare(joined, holdout_fraction=0.5)
    assert set(estimates) == {"observed", "ipw"}
    assert estimates["observed"].rows == 3
    assert estimates["ipw"].rows == 6
    # The declined rows carry weight 2, so the corrected default rate is higher.
    assert estimates["ipw"].positive_rate > estimates["observed"].positive_rate


def test_bootstrap_reports_a_spread_not_a_point():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    probabilities = rng.uniform(0, 1, 500)
    error = bootstrap_std_error(labels, probabilities, np.ones(500), draws=50, seed=0)
    assert error > 0
