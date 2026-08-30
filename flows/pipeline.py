"""Prefect flows for scheduled training and drift response.

The important design decision here is what the automation is *not* allowed to
do. ``retrain_on_drift`` will detect drift, retrain, and register a new
``challenger`` -- but it stops there. Promotion to ``champion`` stays a human
decision, because in a credit-risk setting an unattended pipeline that ships a
model straight to production has no one accountable for the outcome, and drift is
often a data-collection bug rather than a real population shift. Retraining on a
broken feed would faithfully learn the bug.
"""

from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

from credit_default.config import get_settings


@task(retries=2, retry_delay_seconds=30, log_prints=True)
def ingest_task() -> str:
    """Download the source dataset. Retried: the UCI archive is occasionally slow."""
    from credit_default.data.ingest import ingest

    return str(ingest())


@task(log_prints=True)
def split_task() -> dict[str, str]:
    from credit_default.data.split import run

    return {name: str(path) for name, path in run().items()}


@task(log_prints=True)
def train_task() -> dict[str, dict[str, float]]:
    from credit_default.train import run

    return run(register=True)


@task(log_prints=True)
def evaluate_task() -> dict[str, Any]:
    from credit_default.evaluate import evaluate

    return evaluate().to_dict()


@task(log_prints=True)
def drift_task(source: str = "cohort") -> dict[str, Any]:
    from credit_default.monitoring.drift import run

    return run(source)


@flow(name="training", task_runner=ThreadPoolTaskRunner(max_workers=1), log_prints=True)
def training_flow(skip_ingest: bool = False) -> dict[str, Any]:
    """Full training run: data, model, quality gate. Does not promote."""
    logger = get_run_logger()

    if not skip_ingest:
        ingest_task()
    split_task()
    results = train_task()
    gate = evaluate_task()

    if gate["passed"]:
        logger.info(
            "Quality gate passed (PR-AUC %.4f). Promote with 'make promote' after review.",
            gate["challenger"]["pr_auc"],
        )
    else:
        logger.error("Quality gate failed: %s", "; ".join(gate["reasons"]))

    return {"metrics": results, "gate": gate}


@flow(name="drift-check", log_prints=True)
def drift_flow(source: str = "cohort") -> dict[str, Any]:
    """Standalone drift report."""
    logger = get_run_logger()
    summary = drift_task(source)

    drifted = sorted(n for n, i in summary["columns"].items() if i["drifted"])
    if summary["passed"]:
        logger.info("No significant drift (share %.3f).", summary["drifted_share"])
    else:
        logger.warning(
            "Drift detected in %d/%d columns (share %.3f): %s",
            summary["drifted_columns"],
            summary["monitored_columns"],
            summary["drifted_share"],
            ", ".join(drifted),
        )
    return summary


@flow(name="retrain-on-drift", log_prints=True)
def retrain_on_drift(source: str = "cohort", force: bool = False) -> dict[str, Any]:
    """Retrain only when drift warrants it, and stop short of promoting.

    Returns a summary describing what happened, including the explicit manual
    step required to actually put a new model in front of traffic.
    """
    logger = get_run_logger()
    settings = get_settings()

    summary = drift_flow(source)
    if summary["passed"] and not force:
        logger.info("No drift beyond threshold; skipping retraining.")
        return {"retrained": False, "drift": summary}

    logger.warning("Retraining because drift exceeded the threshold.")
    results = train_task()
    gate = evaluate_task()

    if gate["passed"]:
        logger.info(
            "A new @%s is registered and passed the quality gate. "
            "Promotion to @%s is deliberately manual -- review the drift report first, "
            "then run 'make promote'.",
            settings.challenger_alias,
            settings.champion_alias,
        )
    else:
        logger.error(
            "The retrained model failed the quality gate, so it must not be promoted: %s",
            "; ".join(gate["reasons"]),
        )

    return {"retrained": True, "drift": summary, "metrics": results, "gate": gate}


if __name__ == "__main__":
    training_flow()
