"""Train candidate models and record everything in MLflow.

Two models are trained so the registry has something to compare: an interpretable
logistic-regression baseline and a gradient-boosted challenger. The better model
by PR-AUC is registered and tagged with the ``challenger`` alias; promotion to
``champion`` is a separate, deliberate step (see ``promote.py``).

Protected attributes (see ``config.PROTECTED_ATTRIBUTES``) are excluded from the
feature set by default, because sex and marital status are prohibited bases for a
credit decision under ECOA. They remain in the input frame so that fairness can
still be audited across those groups -- excluding an attribute does not remove
its influence, since other features correlate with it, and the only way to know
whether exclusion helped is to measure.

A note on the class imbalance (~22% positive): no resampling or ``class_weight``
rebalancing is applied. Rebalancing would improve headline recall but distort the
predicted probabilities, and a credit-risk score is only useful if its
probabilities mean something. Instead the imbalance is handled by *evaluating*
with PR-AUC rather than accuracy, and calibration is logged as evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in CI or containers

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_default.config import (
    AUDIT_ATTRIBUTES,
    FEATURES,
    PROTECTED_ATTRIBUTES,
    TARGET,
    Settings,
    get_settings,
    model_features,
)
from credit_default.fairness import audit, flatten_for_mlflow
from credit_default.features.pipeline import build_pipeline
from credit_default.threshold import tune_on_training_data

logger = logging.getLogger(__name__)

# MLflow 3 serialises sklearn models with skops, which refuses to persist types it
# does not recognise. Rather than downgrade to pickle, the XGBoost types the
# pipeline legitimately contains are declared trusted explicitly -- that keeps the
# safety check active for everything else.
SKOPS_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]

# Hand-picked starting point, used when no search has been run. scripts/tune.py
# reports against these so the value of tuning is measured, not assumed.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


@dataclass
class Candidate:
    name: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)


def _tuned_params(settings: Settings) -> dict[str, Any] | None:
    """Load parameters from a completed search, if one has been run.

    Kept optional on purpose: the pipeline must stay runnable from a clean clone
    without first spending several minutes on a search.
    """
    path = settings.reports_dir / "best_params.json"
    if not path.exists():
        return None
    try:
        params = json.loads(path.read_text())["best_params"]
        logger.info("Using tuned hyperparameters from %s", path)
        return dict(params)
    except (KeyError, ValueError):
        logger.warning("Ignoring malformed %s", path)
        return None


def candidates(seed: int, tuned: dict[str, Any] | None = None) -> list[Candidate]:
    return [
        Candidate(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, random_state=seed),
            params={"model_family": "logistic_regression", "max_iter": 1000},
        ),
        Candidate(
            name="xgboost",
            estimator=XGBClassifier(
                **(tuned or DEFAULT_XGB_PARAMS),
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=-1,
            ),
            params={
                "model_family": "xgboost",
                "tuned": tuned is not None,
                **(tuned or DEFAULT_XGB_PARAMS),
            },
        ),
    ]


def compute_metrics(
    y_true: pd.Series, probabilities: Any, threshold: float = 0.5
) -> dict[str, float]:
    """PR-AUC leads: with a 22% positive rate, accuracy is not informative.

    Threshold-free metrics (PR-AUC, ROC-AUC, Brier) judge the model; the
    confusion counts and F1 depend on where the cutoff is placed.
    """
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def _plot_diagnostics(y_true: pd.Series, probabilities: Any, output_dir: Path) -> list[Path]:
    """Precision-recall and calibration curves, saved as run artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.axhline(float(y_true.mean()), ls="--", c="grey", label="no-skill baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    pr_path = output_dir / "precision_recall.png"
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    paths.append(pr_path)

    true_frac, pred_frac = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pred_frac, true_frac, marker="o", label="model")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Calibration")
    ax.legend()
    fig.tight_layout()
    cal_path = output_dir / "calibration.png"
    fig.savefig(cal_path, dpi=120)
    plt.close(fig)
    paths.append(cal_path)

    return paths


def train_candidate(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    settings: Settings,
) -> tuple[Pipeline, dict[str, float]]:
    numeric, categorical = model_features(settings.use_protected_attributes)
    pipeline = build_pipeline(candidate.estimator, numeric, categorical)

    with mlflow.start_run(run_name=candidate.name):
        mlflow.log_params(candidate.params)
        mlflow.log_params(
            {
                "train_rows": len(train),
                "test_rows": len(test),
                "seed": settings.random_seed,
                "uses_protected_attributes": settings.use_protected_attributes,
                "excluded_attributes": ",".join(
                    [] if settings.use_protected_attributes else PROTECTED_ATTRIBUTES
                ),
            }
        )

        # Tuned before the final fit, on out-of-fold training predictions, so the
        # cutoff never sees the test set it is later judged on.
        choice = tune_on_training_data(
            pipeline,
            train,
            FEATURES,
            TARGET,
            settings.cost_false_negative,
            settings.cost_false_positive,
            settings.random_seed,
        )
        mlflow.log_metrics(choice.to_dict())
        mlflow.log_params(
            {
                "cost_false_negative": settings.cost_false_negative,
                "cost_false_positive": settings.cost_false_positive,
            }
        )

        pipeline.fit(train[FEATURES], train[TARGET])
        probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
        metrics = compute_metrics(test[TARGET], probabilities, choice.threshold)
        mlflow.log_metrics(metrics)

        # Audited on every run, whether or not the attributes were used as
        # features -- a fairness number you only look at once is not monitoring.
        audits = audit(test, test[TARGET], probabilities, AUDIT_ATTRIBUTES, choice.threshold)
        mlflow.log_metrics(flatten_for_mlflow(audits))
        mlflow.log_dict({name: a.to_dict() for name, a in audits.items()}, "fairness/audit.json")

        for path in _plot_diagnostics(
            test[TARGET], probabilities, settings.reports_dir / candidate.name
        ):
            mlflow.log_artifact(str(path), artifact_path="diagnostics")

        # The signature is what lets the serving layer reject malformed input at
        # the model boundary as well as at the API boundary.
        signature = infer_signature(test[FEATURES], probabilities)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=test[FEATURES].head(5),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            # Travels with the artifact so the serving layer uses the cutoff this
            # model was tuned for, rather than a constant compiled into the API.
            metadata={"decision_threshold": choice.threshold},
        )
        mlflow.set_tag("candidate", candidate.name)

        logger.info(
            "%-20s pr_auc=%.4f roc_auc=%.4f brier=%.4f",
            candidate.name,
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["brier_score"],
        )
    metrics["decision_threshold"] = choice.threshold
    return pipeline, metrics


def register_best(
    best_name: str,
    best_pipeline: Pipeline,
    best_metrics: dict[str, float],
    test: pd.DataFrame,
    settings: Settings,
) -> None:
    """Register the winner and mark it ``challenger`` -- never ``champion``.

    The tuned threshold is attached here as well as on the candidate run. Every
    persisted copy of a model must carry its own cutoff; a copy without one
    silently falls back to 0.5, which is the bug this replaced.
    """
    signature = infer_signature(test[FEATURES], best_pipeline.predict_proba(test[FEATURES])[:, 1])
    with mlflow.start_run(run_name=f"register-{best_name}"):
        mlflow.log_metrics(best_metrics)
        mlflow.set_tag("candidate", best_name)
        info = mlflow.sklearn.log_model(
            best_pipeline,
            name="model",
            signature=signature,
            input_example=test[FEATURES].head(5),
            registered_model_name=settings.registered_model_name,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            metadata={"decision_threshold": best_metrics["decision_threshold"]},
        )

    client = MlflowClient()
    newest = client.search_model_versions(
        f"name='{settings.registered_model_name}'",
        order_by=["version_number DESC"],
        max_results=1,
    )
    version = newest[0].version
    client.set_registered_model_alias(
        settings.registered_model_name, settings.challenger_alias, version
    )
    logger.info(
        "Registered %s v%s as @%s (uri=%s)",
        settings.registered_model_name,
        version,
        settings.challenger_alias,
        info.model_uri,
    )


def run(register: bool = True) -> dict[str, dict[str, float]]:
    settings = get_settings()
    settings.ensure_dirs()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    train = pd.read_parquet(settings.train_parquet)
    test = pd.read_parquet(settings.test_parquet)

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for candidate in candidates(settings.random_seed, _tuned_params(settings)):
        pipeline, metrics = train_candidate(candidate, train, test, settings)
        results[candidate.name] = metrics
        fitted[candidate.name] = pipeline

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    logger.info("Best candidate: %s (pr_auc=%.4f)", best_name, results[best_name]["pr_auc"])

    # Always persist locally so the API can run with no MLflow server reachable.
    # save_model refuses to write into a non-empty directory, so reruns need a clean slate.
    if settings.local_model_path.exists():
        shutil.rmtree(settings.local_model_path)
    mlflow.sklearn.save_model(
        fitted[best_name],
        str(settings.local_model_path),
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
        metadata={"decision_threshold": results[best_name]["decision_threshold"]},
    )

    settings.metrics_path.write_text(
        json.dumps({"best": best_name, "candidates": results}, indent=2)
    )

    if register:
        register_best(best_name, fitted[best_name], results[best_name], test, settings)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register candidate models.")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Train and log runs without touching the model registry.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    run(register=not args.no_register)


if __name__ == "__main__":
    main()
