"""Linear and generalized linear models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import (
    LinearRegression,
)
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor
from loyalty_spend.models.registry import register

ALPHAS = np.logspace(-4, 3, 30)


def _pipeline(model, encoding: str = "onehot") -> Pipeline:
    return Pipeline([("prep", make_preprocessor(encoding)), ("model", model)])


@register("ols", family="linear")
def ols() -> Pipeline:
    """Ordinary least squares on one-hot categories; the baseline."""
    return _pipeline(LinearRegression())
