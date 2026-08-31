#!/usr/bin/env python
"""Run the hyperparameter search and report whether it was worth it.

Prints the comparison against the hand-picked baseline and writes the winning
parameters to reports/best_params.json, which train.py picks up automatically.

A search that finds nothing is still a useful result. Reporting "tuning bought
0.002 PR-AUC" is more honest, and more informative, than quietly shipping the
tuned model and implying the search mattered.
"""

from __future__ import annotations

import argparse
import json
import logging

import mlflow
import pandas as pd

from credit_default.config import get_settings
from credit_default.tuning import search

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    settings.ensure_dirs()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    train = pd.read_parquet(settings.train_parquet)

    with mlflow.start_run(run_name="hyperparameter-search"):
        mlflow.log_params({"n_trials": args.trials, "cv_folds": args.folds})
        result = search(train, settings, n_trials=args.trials, folds=args.folds)
        mlflow.log_metrics(
            {
                "best_cv_pr_auc": result.best_score,
                "baseline_cv_pr_auc": result.baseline_score,
                "improvement": result.improvement,
            }
        )
        mlflow.log_dict(result.to_dict(), "tuning/result.json")

    print("\n" + "=" * 70)
    print(f"{'HYPERPARAMETER SEARCH':^70}")
    print("=" * 70)
    print(f"  trials                 {result.n_trials}")
    print(f"  hand-picked baseline   {result.baseline_score:.5f}  CV PR-AUC")
    print(f"  best found             {result.best_score:.5f}  CV PR-AUC")
    print(f"  improvement            {result.improvement:+.5f}")
    print()
    if result.improvement < 0.002:
        print("  The search found little. The hand-picked parameters were already")
        print("  close to the best this model family can do on this data, which is")
        print("  worth knowing and worth saying.")
    print("\n  best parameters:")
    for k, v in sorted(result.best_params.items()):
        print(f"    {k:20} {v}")

    destination = settings.reports_dir / "best_params.json"
    destination.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\n  written to {destination}")


if __name__ == "__main__":
    main()
