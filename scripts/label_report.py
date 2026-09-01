#!/usr/bin/env python
"""Measure what late, censored labels actually do to a performance number.

Four questions, each answered with a number rather than a paragraph:

1. **How wrong is a join that ignores time?** The naive join is run alongside the
   point-in-time-correct one and the gap is reported. If the answer were "barely
   any", the whole apparatus would not be worth its weight -- so it is measured.

2. **What does an immature cohort look like?** Performance is recomputed at a
   series of as-of dates. Because defaults are reported more slowly than
   non-defaults, an early evaluation should understate the default rate and
   flatter the model, and the size of that distortion is the point.

3. **How much does censoring bias production metrics?** The model chose which
   applicants ever get a label. The naive observed number, the IPW-corrected
   number and the true full-population number are reported side by side, so the
   correction can be checked against an answer key rather than trusted.

4. **What does the holdout cost, and how small can it be?** Sweeping the holdout
   rate trades precision against knowingly approving riskier applicants. Both
   sides are quantified, in the same cost units the decision threshold uses.

    python scripts/label_report.py
    python scripts/label_report.py --source postgres
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from credit_default.api.model import load_model
from credit_default.config import TARGET, get_settings
from credit_default.labels import performance
from credit_default.labels.backfill import (
    join_labels,
    join_labels_ignoring_time,
    load_served_predictions,
)
from credit_default.labels.store import build_outcome_store

logger = logging.getLogger(__name__)

# Days after the first decision at which to re-evaluate. The cohort spans 180
# days, so anything below that is a partially-originated book as well as a
# partially-matured one -- which is exactly the situation a real monitoring job
# finds itself in every day of its life.
AS_OF_OFFSETS = [90, 150, 180, 210, 240, 270, 330, 400]
HOLDOUT_SWEEP = [0.01, 0.02, 0.05, 0.10, 0.25]
# Independent draws per rate. The estimate's spread is the whole point of the
# sweep, and it cannot be seen from one sample.
HOLDOUT_REPEATS = 25
# Above this standard error, or below this many effective rows, the corrected
# estimate is reported with an explicit warning instead of as a result.
IPW_UNUSABLE_STD_ERROR = 0.15
IPW_MIN_EFFECTIVE_ROWS = 100.0
RULE = "=" * 88


def _bar(value: float, width: int = 24) -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return "#" * filled + "." * (width - filled)


def _pr_auc(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray | None) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities, sample_weight=weights))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["parquet", "postgres"], default="parquet")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    settings = get_settings()
    settings.ensure_dirs()

    served = load_served_predictions(settings, args.source)
    served["predicted_at"] = pd.to_datetime(served["predicted_at"], utc=True)
    outcomes = build_outcome_store(settings).read()
    window = settings.label_performance_window_days
    holdout = settings.label_holdout_fraction

    # The answer key belongs to the replay and to nothing else. Loading it
    # alongside real served traffic would compare a live cohort against the
    # ground truth of a different one and report the difference as a finding --
    # the precise class of quiet wrongness this whole report exists to catch.
    # Production has no answer key, so sections 3 and 4 degrade rather than lie.
    truth_path = settings.processed_dir / "label_truth.parquet"
    truth = (
        pd.read_parquet(truth_path) if args.source == "parquet" and truth_path.exists() else None
    )
    if truth is None:
        print(
            "\nNo answer key for this source: the full-population estimate and the "
            "holdout sweep\nare omitted. That is the normal production case -- a declined "
            "applicant's outcome\ndoes not exist, so there is nothing to check the "
            "correction against."
        )

    origin = served["predicted_at"].min()
    fully_matured = origin + pd.Timedelta(float(AS_OF_OFFSETS[-1]), unit="D")
    payload: dict[str, object] = {
        "source": args.source,
        "decisions": len(served),
        "performance_window_days": window,
        "holdout_fraction": holdout,
        "origin": origin.isoformat(),
    }

    # ---------------------------------------------------------------- 1 ------
    # Chosen so that both gates are actually doing work: the book is still being
    # originated (so some windows have not closed) and the slow tail of defaults
    # has not landed (so some closed windows are still unreported). A date where
    # only one gate bites would understate the problem.
    probe = origin + pd.Timedelta(150.0, unit="D")
    correct = join_labels(served, outcomes, probe.to_pydatetime(), window)
    naive = join_labels_ignoring_time(served, outcomes)

    correct_auc = _pr_auc(
        correct.frame["label"].to_numpy(), correct.frame["probability"].to_numpy(), None
    )
    naive_auc = _pr_auc(naive["label"].to_numpy(), naive["probability"].to_numpy(), None)

    print("\n" + RULE)
    print(f"1. POINT-IN-TIME CORRECTNESS   (evaluating as of {probe.date()})")
    print(RULE)
    print(f"{'':<34}{'rows scored':>14}{'PR-AUC':>12}{'positive rate':>16}")
    print(
        f"{'point-in-time correct join':<34}{correct.observed:>14,}{correct_auc:>12.4f}"
        f"{correct.frame['label'].mean():>16.4f}"
    )
    print(
        f"{'naive join (ignores time)':<34}{len(naive):>14,}{naive_auc:>12.4f}"
        f"{naive['label'].mean():>16.4f}"
    )
    print(
        f"{'imported from the future':<34}{len(naive) - correct.observed:>14,}"
        f"{naive_auc - correct_auc:>+12.4f}"
    )
    print(
        f"\n  Of {correct.total:,} decisions, {correct.matured:,} had matured "
        f"({correct.maturity:.1%}) and {correct.observed:,} had been reported "
        f"(coverage {correct.coverage:.1%})."
    )
    print(
        f"  {correct.pending:,} approved outcomes were still in the post; "
        f"{correct.censored:,} were declined and will never exist."
    )
    payload["point_in_time"] = {
        "as_of": probe.isoformat(),
        "correct": correct.to_dict() | {"pr_auc": round(correct_auc, 4)},
        "naive_rows": len(naive),
        "naive_pr_auc": round(naive_auc, 4),
        "rows_from_the_future": len(naive) - correct.observed,
    }

    # ---------------------------------------------------------------- 2 ------
    print("\n" + RULE)
    print("2. WHAT AN IMMATURE COHORT LOOKS LIKE")
    print(RULE)
    # Deliberately the *naive* approved-only numbers: this section is about what a
    # monitoring dashboard would actually display as a cohort matures, and no
    # dashboard is reweighting anything. The correction belongs in section 3.
    print(
        f"{'as of':<14}{'matured':>9}{'coverage':>10}{'labels':>9}"
        f"{'obs. default rate':>19}{'PR-AUC':>9}   coverage"
    )
    curve = []
    for offset in AS_OF_OFFSETS:
        as_of = origin + pd.Timedelta(float(offset), unit="D")
        join = join_labels(served, outcomes, as_of.to_pydatetime(), window)
        if join.observed == 0:
            continue
        approved = join.frame[join.frame["prediction"] == 0]
        labels = approved["label"].to_numpy()
        row = {
            "as_of": as_of.isoformat(),
            "days_after_first_decision": offset,
            "maturity": round(join.maturity, 4),
            "coverage": round(join.coverage, 4),
            "labels": join.observed,
            "observed_default_rate": round(float(labels.mean()), 4),
            "observed_pr_auc": round(_pr_auc(labels, approved["probability"].to_numpy(), None), 4),
        }
        curve.append(row)
        print(
            f"{as_of.date()!s:<14}{join.maturity:>8.0%}{join.coverage:>10.0%}"
            f"{join.observed:>9,}{labels.mean():>19.4f}{row['observed_pr_auc']:>9.4f}"
            f"   {_bar(join.coverage)}"
        )
    payload["maturity_curve"] = curve

    if curve:
        early, late = curve[0], curve[-1]
        drift_like = late["observed_default_rate"] - early["observed_default_rate"]
        print(
            f"\n  The observed default rate moves {drift_like:+.4f} between "
            f"{early['days_after_first_decision']} and {late['days_after_first_decision']} days "
            "purely as slow files land."
        )
        print(
            "  Nothing about the model or the population changed. A dashboard "
            "comparing today's\n  observed rate against last month's would report that as drift."
        )

    # ---------------------------------------------------------------- 3 ------
    final = join_labels(served, outcomes, fully_matured.to_pydatetime(), window)
    estimates = performance.compare(final.frame, holdout, truth, settings.random_seed)

    weights = 1.0 / performance.selection_probability(final.frame["prediction"], holdout)
    ipw_se = performance.bootstrap_std_error(
        final.frame["label"].to_numpy(),
        final.frame["probability"].to_numpy(),
        weights,
        seed=settings.random_seed,
    )

    # The right yardstick for the observed number is not the headline offline
    # PR-AUC -- it is the offline PR-AUC recomputed over the same approved
    # subpopulation. Comparing against anything else guarantees a false alarm.
    handle = load_model(settings)
    test = pd.read_parquet(settings.test_parquet)
    test_probabilities, test_decisions = handle.predict(test)
    test_probabilities = np.asarray(test_probabilities)
    approved_mask = np.asarray(test_decisions) == 0
    offline_full = _pr_auc(test[TARGET].to_numpy(), test_probabilities, None)
    offline_approved = _pr_auc(
        test[TARGET].to_numpy()[approved_mask], test_probabilities[approved_mask], None
    )

    print("\n" + RULE)
    print(f"3. CENSORING   (fully matured, as of {fully_matured.date()})")
    print(RULE)
    print(f"{'estimate':<48}{'rows':>9}{'eff. rows':>11}{'PR-AUC':>10}")
    for key in ("observed", "ipw", "full"):
        if key not in estimates:
            continue
        e = estimates[key]
        suffix = f"  +/- {ipw_se:.4f}" if key == "ipw" else ""
        print(f"{e.name:<48}{e.rows:>9,}{e.effective_rows:>11,.0f}{e.pr_auc:>10.4f}{suffix}")
    print(
        f"\n{'offline test PR-AUC, full population':<48}{len(test):>9,}{'':>11}{offline_full:>10.4f}"
    )
    print(
        f"{'offline test PR-AUC, approved subpopulation':<48}"
        f"{int(approved_mask.sum()):>9,}{'':>11}{offline_approved:>10.4f}"
    )
    # A correction that is unbiased but wildly imprecise is not a usable number,
    # and it looks exactly like a usable one on the page. Small cohorts and small
    # holdouts both land here: one holdout row reweighted by 50 can swing the
    # estimate by half its range. Say so, rather than letting the figure be quoted.
    if ipw_se > IPW_UNUSABLE_STD_ERROR or estimates["ipw"].effective_rows < IPW_MIN_EFFECTIVE_ROWS:
        print(
            f"\n  WARNING: the IPW estimate rests on {estimates['ipw'].effective_rows:.0f} "
            f"effective rows and has a standard\n  error of {ipw_se:.4f}. It is too imprecise "
            "to support a conclusion -- treat it as\n  evidence that the holdout or the cohort "
            "is too small, not as a performance number."
        )

    print(
        f"\n  The observed number is {estimates['observed'].pr_auc - offline_full:+.4f} against the "
        f"headline offline PR-AUC\n  but {estimates['observed'].pr_auc - offline_approved:+.4f} "
        "against the only comparison that is actually like-for-like."
    )
    payload["censoring"] = {key: e.to_dict() for key, e in estimates.items()} | {
        "ipw_bootstrap_std_error": round(ipw_se, 4),
        "offline_test_pr_auc_full": round(offline_full, 4),
        "offline_test_pr_auc_approved": round(offline_approved, 4),
        "join": final.to_dict(),
    }

    # ---------------------------------------------------------------- 4 ------
    if truth is not None:
        print("\n" + RULE)
        print("4. WHAT THE HOLDOUT BUYS, AND WHAT IT COSTS")
        print(RULE)
        print(
            f"{'holdout':>9}{'extra approvals':>17}{'IPW PR-AUC':>13}{'spread':>11}"
            f"{'error vs truth':>16}{'cost / 1k applicants':>22}"
        )
        sweep = []
        declined = truth["prediction"].to_numpy() == 1
        true_auc = estimates["full"].pr_auc if "full" in estimates else float("nan")
        for fraction in HOLDOUT_SWEEP:
            # Repeated draws, because a single holdout sample says almost nothing
            # at these rates: the spread across draws IS the finding, and one
            # realisation would show a non-monotone error column that looks like
            # a bug rather than like sampling noise.
            aucs, costs, extras = [], [], []
            for repeat in range(HOLDOUT_REPEATS):
                rng = np.random.default_rng(settings.random_seed + repeat)
                selected = ~declined | (declined & (rng.random(len(truth)) < fraction))
                sample = truth[selected]
                w = 1.0 / performance.selection_probability(sample["prediction"], fraction)
                auc = _pr_auc(sample["label"].to_numpy(), sample["probability"].to_numpy(), w)
                if not np.isnan(auc):
                    aucs.append(auc)
                # Every holdout approval swaps a would-be false positive for a
                # possible false negative, priced with the same ratio the decision
                # threshold is derived from.
                in_holdout = truth[declined & selected]["label"].to_numpy()
                extras.append(int(in_holdout.size))
                costs.append(
                    (
                        settings.cost_false_negative * int(in_holdout.sum())
                        - settings.cost_false_positive * int((in_holdout == 0).sum())
                    )
                    / len(truth)
                    * 1000
                )

            mean_auc = float(np.mean(aucs))
            spread = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")
            sweep.append(
                {
                    "holdout_fraction": fraction,
                    "draws": HOLDOUT_REPEATS,
                    "mean_extra_approvals": round(float(np.mean(extras)), 1),
                    "mean_pr_auc_ipw": round(mean_auc, 4),
                    "std_across_draws": round(spread, 4),
                    "mean_error_vs_truth": round(mean_auc - true_auc, 4),
                    "cost_per_1000_applicants": round(float(np.mean(costs)), 2),
                }
            )
            print(
                f"{fraction:>9.0%}{float(np.mean(extras)):>17,.0f}{mean_auc:>13.4f}{spread:>11.4f}"
                f"{mean_auc - true_auc:>+16.4f}{float(np.mean(costs)):>22.1f}"
            )
        payload["holdout_sweep"] = sweep
        print(
            f"\n  Mean of {HOLDOUT_REPEATS} independent draws per rate; 'spread' is the standard "
            f"deviation across them.\n  True full-population PR-AUC is {true_auc:.4f}. Cost is in "
            "the same units as the\n"
            f"  decision threshold ({settings.cost_false_negative:.0f}:1 false negative to false "
            "positive), so it is\n  comparable to the expected-cost figures in the threshold report."
        )

    destination = args.output or (settings.reports_dir / "label_performance.json")
    with open(destination, "w") as handle_out:  # noqa: PTH123
        json.dump(payload, handle_out, indent=2, default=str)
    print(f"\nWritten to {destination}")


if __name__ == "__main__":
    main()
