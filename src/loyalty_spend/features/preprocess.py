"""Feature preprocessing shared by all scikit-learn models."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

from loyalty_spend.data.schema import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
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


def make_preprocessor(kind: str = "onehot") -> ColumnTransformer:
    """Column transformer for one of three encodings.

    ``onehot``
        Standardized numerics and one-hot categories; for linear models.
    ``ordinal``
        Integer-coded categories; for tree ensembles, which split on codes.
    ``spline``
        Cubic B-spline basis of the numerics and one-hot categories; an
        additive model with smooth effects when followed by a linear model.
    """
    if kind not in ENCODINGS:
        raise ValueError(f"kind must be one of {ENCODINGS}")
    if kind == "ordinal":
        cat = OrdinalEncoder(
            categories=_categories(),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        num = SimpleImputer(strategy="median", add_indicator=True)
    else:
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                (
                    "onehot",
                    OneHotEncoder(
                        categories=_categories(),
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        num = _numeric(kind)
    return ColumnTransformer(
        [("num", num, NUMERIC), ("cat", cat, CATEGORICAL)],
        verbose_feature_names_out=False,
    )
