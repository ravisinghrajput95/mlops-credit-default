"""Central, environment-driven configuration.

Every path, URI and threshold in the project resolves through this module so that
the same code runs unchanged locally, in CI, and on Cloud Run -- only env vars differ.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The UCI archive serves the dataset as a zipped legacy .xls.
DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

TARGET = "default_payment_next_month"

# PAY_* are ordinal repayment-status codes, not quantities, so they are treated as
# categorical downstream even though they are stored as integers.
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class Settings(BaseSettings):
    """Runtime settings, overridable by env var or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- data splitting ----------------------------------------------------
    random_seed: int = 42
    # Held out to stand in for live production traffic during the drift demo.
    current_fraction: float = 0.25
    test_fraction: float = 0.2

    # --- mlflow ------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "credit-default"
    registered_model_name: str = "credit-default-classifier"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"

    # --- model quality gate ------------------------------------------------
    # The class balance is ~22% positive, so PR-AUC is the primary metric;
    # accuracy would be ~78% for a model that never predicts default.
    min_pr_auc: float = 0.50
    # How much worse than the incumbent champion a challenger may be before
    # evaluate.py fails the build.
    max_pr_auc_regression: float = 0.01

    # --- drift -------------------------------------------------------------
    # Share of monitored columns that may drift before the report fails.
    max_drifted_share: float = 0.3

    # --- serving -----------------------------------------------------------
    api_host: str = "0.0.0.0"  # bound inside a container
    api_port: int = 8000
    # "registry" pulls the aliased model from MLflow; "local" loads from disk so
    # the API can start with no MLflow server reachable (CI, offline demo).
    model_source: Literal["registry", "local"] = "local"
    local_model_path: Path = PROJECT_ROOT / "data" / "models" / "model"

    # --- prediction sink ---------------------------------------------------
    # Postgres locally (docker-compose); GCS Parquet in the cloud, which keeps
    # the deployment inside GCP's always-free tier. See infra/gcp.
    prediction_sink: Literal["postgres", "gcs", "none"] = "none"
    postgres_dsn: str = "postgresql://mlops:mlops@localhost:5432/mlops"
    gcs_prediction_prefix: str = ""

    log_level: str = Field(default="INFO")

    # --- derived paths -----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def raw_parquet(self) -> Path:
        return self.raw_dir / "credit_default.parquet"

    @property
    def reference_parquet(self) -> Path:
        return self.processed_dir / "reference.parquet"

    @property
    def current_parquet(self) -> Path:
        return self.processed_dir / "current.parquet"

    @property
    def train_parquet(self) -> Path:
        return self.processed_dir / "train.parquet"

    @property
    def test_parquet(self) -> Path:
        return self.processed_dir / "test.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics.json"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.models_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton; call this rather than instantiating Settings."""
    return Settings()
