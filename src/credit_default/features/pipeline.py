"""Feature preprocessing, fused into the model artifact.

The preprocessing lives *inside* the fitted sklearn Pipeline rather than as a
separate step run before serving. That is deliberate: it makes the artifact
self-contained, so the transformation applied at inference is by construction the
same object that was fitted at training time. Train/serve skew from a
reimplemented or drifted preprocessing step is the single most common way a
model that scored well offline degrades silently in production.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.config import model_features


def build_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """Scale numerics, one-hot the ordinal-coded categoricals.

    ``handle_unknown="ignore"`` means an unseen category at inference time yields
    an all-zero block instead of raising -- the API should degrade rather than
    500 when production sends a code the training data never contained.
    """
    if numeric is None or categorical is None:
        numeric, categorical = model_features()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                categorical,
            ),
        ],
        remainder="drop",  # never let an unexpected column reach the estimator
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator: Any,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> Pipeline:
    """Fuse preprocessing and estimator into one fit/predict unit.

    ``remainder="drop"`` matters more than usual here: protected attributes stay
    in the input frame for auditing, and this is what guarantees the estimator
    never actually sees them.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("estimator", estimator),
        ]
    )
