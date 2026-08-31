"""The decision threshold must follow business cost, not convention."""

from __future__ import annotations

import numpy as np
import pytest

from credit_default.threshold import choose_threshold, expected_cost


@pytest.fixture
def scores():
    """Perfectly calibrated scores: an outcome drawn with its own probability.

    Calibration matters for these tests. For a calibrated model the cost-optimal
    cutoff has a closed form -- ``c_fp / (c_fn + c_fp)`` -- so the search can be
    checked against theory rather than against another run of itself.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 20_000)
    y = (rng.uniform(0.0, 1.0, 20_000) < p).astype(int)
    return y, p


def analytical_optimum(cost_false_negative: float, cost_false_positive: float) -> float:
    return cost_false_positive / (cost_false_negative + cost_false_positive)


@pytest.mark.parametrize(
    ("cost_fn", "cost_fp"),
    [(1.0, 1.0), (5.0, 1.0), (10.0, 1.0), (1.0, 5.0), (1.0, 10.0)],
)
def test_search_finds_the_analytically_optimal_cutoff(scores, cost_fn, cost_fp):
    """The whole point: the cutoff follows the cost ratio, not convention."""
    y, p = scores
    choice = choose_threshold(y, p, cost_fn, cost_fp)
    # Tolerance is 0.05 because the expected-cost curve is genuinely flat near its
    # minimum -- at the 1:1 ratio, cutoffs of 0.47 and 0.50 differ in cost by about
    # 0.001. That flatness is a useful property, not noise to be tuned away: it
    # means the exact cutoff matters far less than the cost ratio behind it.
    assert choice.threshold == pytest.approx(analytical_optimum(cost_fn, cost_fp), abs=0.05)


def test_equal_costs_are_the_only_case_where_half_is_right(scores):
    """0.5 is not a neutral default; it encodes a specific cost assumption."""
    y, p = scores
    assert choose_threshold(y, p, 1.0, 1.0).threshold == pytest.approx(0.5, abs=0.05)


def test_expensive_false_negatives_lower_the_threshold(scores):
    """If missing a defaulter hurts more, decline more readily."""
    y, p = scores
    assert choose_threshold(y, p, 10.0, 1.0).threshold < choose_threshold(y, p, 1.0, 1.0).threshold


def test_expensive_false_positives_raise_the_threshold(scores):
    """The converse: if wrongly declining is costly, demand more evidence."""
    y, p = scores
    assert choose_threshold(y, p, 1.0, 10.0).threshold > choose_threshold(y, p, 1.0, 1.0).threshold


def test_the_chosen_threshold_is_at_least_as_good_as_one_half(scores):
    """Whatever it picks must not be worse than the default it replaces."""
    y, p = scores
    choice = choose_threshold(y, p, 5.0, 1.0)
    assert choice.expected_cost <= choice.cost_at_half


def test_expected_cost_counts_both_error_types():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.9, 0.1])  # one FP and one FN at 0.5
    # 1 false negative at cost 5, 1 false positive at cost 1, over 4 applicants.
    assert expected_cost(y, p, 0.5, 5.0, 1.0) == pytest.approx((5.0 + 1.0) / 4)


def test_a_perfect_model_has_zero_cost():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.1, 0.9, 1.0])
    assert expected_cost(y, p, 0.5, 5.0, 1.0) == 0.0


def test_choice_serialises_for_tracking(scores):
    y, p = scores
    payload = choose_threshold(y, p, 5.0, 1.0).to_dict()
    assert payload["decision_threshold"] > 0
    assert payload["cost_reduction_vs_0.5"] >= 0
