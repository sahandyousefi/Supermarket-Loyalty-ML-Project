r"""Feasible weighted least squares with an error variance per group.

The out-of-fold residuals of OLS have standard deviations from 0.10 to
0.55 across joining months. OLS stays unbiased under such
heteroscedasticity but is not efficient, and a single residual variance
misstates the uncertainty of every member. Feasible WLS estimates the
variance of each group from OLS residuals, shrinks it towards the pooled
variance in proportion to the group size,

.. math::

    \hat\sigma_g^2 = \frac{n_g s_g^2 + k s^2}{n_g + k},

refits with weights :math:`1/\hat\sigma_g^2`, and repeats the two steps
until the variances settle. Groups not seen in training get the pooled
variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from loyalty_spend.arrays import FloatArray, ObjectArray
from loyalty_spend.features.preprocess import make_preprocessor
from loyalty_spend.models.registry import register


class GroupWeightedRegression(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        group: str = "joining_month",
        shrinkage: float = 10.0,
        n_iter: int = 3,
        estimator: BaseEstimator | None = None,
    ) -> None:
        self.group = group
        self.shrinkage = shrinkage
        self.n_iter = n_iter
        self.estimator = estimator

    def _base(self) -> Pipeline:
        if self.estimator is not None:
            return clone(self.estimator)
        return Pipeline(
            [("prep", make_preprocessor("onehot")), ("model", LinearRegression())]
        )

    def _labels(self, X: pd.DataFrame) -> ObjectArray:
        return np.asarray(X[self.group].astype(str).fillna("<missing>"), dtype=object)

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> GroupWeightedRegression:
        y = np.asarray(y, float)
        labels = self._labels(X)
        weights: FloatArray = np.ones(len(y))
        for _ in range(self.n_iter):
            model = self._base().fit(X, y, model__sample_weight=weights)
            resid = y - model.predict(X)
            pooled = float(np.mean(resid**2))
            var = {}
            for lv in np.unique(labels):
                r = resid[labels == lv]
                var[lv] = (len(r) * np.mean(r**2) + self.shrinkage * pooled) / (
                    len(r) + self.shrinkage
                )
            weights = 1.0 / np.array([var[lv] for lv in labels])
        self.model_ = model
        self.variance_ = var
        self.pooled_variance_ = pooled
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        return np.asarray(self.model_.predict(X), float)

    def predict_sd(self, X: pd.DataFrame) -> FloatArray:
        """Error standard deviation of each member's group."""
        labels = self._labels(X)
        return np.sqrt(
            np.array([self.variance_.get(lv, self.pooled_variance_) for lv in labels])
        )


@register("wls_month", family="linear")
def wls_month(shrinkage: float = 10.0) -> GroupWeightedRegression:
    """OLS refitted with one error variance per joining month."""
    return GroupWeightedRegression("joining_month", shrinkage)
