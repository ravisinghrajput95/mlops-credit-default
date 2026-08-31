"""Hyperparameter search.

Optimises PR-AUC with Optuna's TPE sampler, which models the relationship between
parameters and score rather than sampling blindly -- it finds a good region in
far fewer trials than grid or random search on a space this size.

Two things are deliberate:

**Search on the training set only.** Every trial is scored by stratified
cross-validation over the training data. The test set is touched exactly once, at
the end, to report the chosen model's performance. Selecting hyperparameters by
test score would make that final number meaningless -- it would be a training
metric wearing a test set's clothes.

**Optimise PR-AUC, not accuracy.** Same reasoning as everywhere else in this
project: at a 22% positive rate, accuracy rewards a model that never predicts
default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from credit_default.config import FEATURES, TARGET, Settings, model_features
from credit_default.features.pipeline import build_pipeline

logger = logging.getLogger(__name__)

# Optuna's own logging is chatty enough to bury the useful output.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_score: float
    baseline_score: float
    n_trials: int

    @property
    def improvement(self) -> float:
        return self.best_score - self.baseline_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_cv_pr_auc": round(self.best_score, 5),
            "baseline_cv_pr_auc": round(self.baseline_score, 5),
            "improvement": round(self.improvement, 5),
            "n_trials": self.n_trials,
            "best_params": self.best_params,
        }


def _suggest(trial: optuna.Trial) -> dict[str, Any]:
    """The search space.

    Ranges are deliberately conservative. On 18k rows with 20 features, deep
    trees and high learning rates overfit quickly, and the aim is a model that
    generalises rather than one that wins the validation fold.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
    }


def _score(params: dict[str, Any], train: pd.DataFrame, settings: Settings, folds: int) -> float:
    numeric, categorical = model_features(settings.use_protected_attributes)
    pipeline = build_pipeline(
        XGBClassifier(
            **params,
            eval_metric="aucpr",
            random_state=settings.random_seed,
            n_jobs=-1,
            verbosity=0,
        ),
        numeric,
        categorical,
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=settings.random_seed)
    scores = cross_val_score(
        pipeline,
        train[FEATURES],
        train[TARGET],
        cv=splitter,
        scoring="average_precision",
        n_jobs=1,
    )
    return float(scores.mean())


def search(
    train: pd.DataFrame,
    settings: Settings,
    n_trials: int = 30,
    folds: int = 3,
    baseline_params: dict[str, Any] | None = None,
) -> SearchResult:
    """Run the search and report it against the hand-picked baseline."""
    baseline_params = baseline_params or {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }

    baseline = _score(baseline_params, train, settings, folds)
    logger.info("Hand-picked baseline: CV PR-AUC %.5f", baseline)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        score = _score(params, train, settings, folds)
        # Each trial is its own nested MLflow run, so the whole search is
        # inspectable afterwards rather than collapsing to a single best number.
        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number:03d}"):
            mlflow.log_params(params)
            mlflow.log_metric("cv_pr_auc", score)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=settings.random_seed),
        study_name="xgboost-pr-auc",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Best trial %d: CV PR-AUC %.5f (%+.5f vs baseline)",
        study.best_trial.number,
        study.best_value,
        study.best_value - baseline,
    )
    return SearchResult(
        best_params=study.best_params,
        best_score=study.best_value,
        baseline_score=baseline,
        n_trials=n_trials,
    )
