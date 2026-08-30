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

from credit_default.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics, one-hot the ordinal-coded categoricals.

    ``handle_unknown="ignore"`` means an unseen category at inference time yields
    an all-zero block instead of raising -- the API should degrade rather than
    500 when production sends a code the training data never contained.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",  # never let an unexpected column reach the estimator
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: Any) -> Pipeline:
    """Fuse preprocessing and estimator into one fit/predict unit."""
    return Pipeline([("preprocessor", build_preprocessor()), ("estimator", estimator)])
