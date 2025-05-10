"""Linear and generalized linear models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    ElasticNetCV,
    GammaRegressor,
    HuberRegressor,
    LinearRegression,
    RidgeCV,
)
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor
from loyalty_spend.models.registry import register

ALPHAS = np.logspace(-4, 3, 30)


def _pipeline(model: BaseEstimator, encoding: str = "onehot") -> Pipeline:
    return Pipeline([("prep", make_preprocessor(encoding)), ("model", model)])


@register("ols", family="linear")
def ols() -> Pipeline:
    """Ordinary least squares on one-hot categories; the baseline."""
    return _pipeline(LinearRegression())


@register("ridge", family="linear")
def ridge(alphas: ArrayLike = ALPHAS) -> Pipeline:
    """Ridge regression with the penalty chosen by efficient leave-one-out CV."""
    return _pipeline(RidgeCV(alphas=alphas))


@register("elastic_net", family="linear")
def elastic_net(
    l1_ratio: Sequence[float] = (0.1, 0.5, 0.9, 1.0), cv: int = 5
) -> Pipeline:
    """Elastic net; the L1 part can drop uninformative dummies."""
    return _pipeline(ElasticNetCV(l1_ratio=list(l1_ratio), cv=cv, max_iter=20000))


@register("huber", family="linear")
def huber(epsilon: float = 1.35, alpha: float = 1e-4) -> Pipeline:
    """Huber loss regression, robust to a minority of extreme residuals."""
    return _pipeline(HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=2000))


@register("gamma_glm", family="linear")
def gamma_glm(alpha: float = 1e-4) -> Pipeline:
    """Gamma GLM with log link.

    Spend is positive and its spread tends to grow with its level; the log
    link makes effects multiplicative and predictions positive.
    """
    return _pipeline(GammaRegressor(alpha=alpha, max_iter=5000))


@register("spline_ridge", family="linear")
def spline_ridge(alphas: ArrayLike = ALPHAS) -> Pipeline:
    """Additive model: cubic B-splines of the numerics, ridge penalty."""
    return _pipeline(RidgeCV(alphas=alphas), encoding="spline")
