"""Feature preprocessing shared by all scikit-learn models."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    SplineTransformer,
    StandardScaler,
)

from loyalty_spend.data.schema import (
    CATEGORICAL,
    FEATURES,
    TARGET,
    allowed_values,
)

ENCODINGS = ("onehot", "ordinal", "spline")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame and target; ids are dropped."""
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found")
    return df[FEATURES].copy(), df[TARGET].astype(float).copy()


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Features of a scoring frame, in model order."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return df[FEATURES].copy()


def _categories() -> list[list[str]]:
    return [allowed_values(c) for c in CATEGORICAL]


def _numeric(kind: str) -> Pipeline:
    # Median imputation with an indicator keeps the information that a value
    # was missing instead of pretending it was zero.
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "spline":
        steps.append(("spline", SplineTransformer(n_knots=6, degree=3)))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)
