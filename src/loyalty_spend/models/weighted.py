r"""Feasible weighted least squares with an error variance per group.

The out-of-fold residuals of OLS have standard deviations from 0.10 to
0.55 across joining months. OLS stays unbiased under such
heteroscedasticity but is not efficient, and a single residual variance
misstates the uncertainty of every member. Feasible WLS estimates the
variance of each group from OLS residuals, shrinks it towards the pooled
variance on the log scale in proportion to the group size,

.. math::

    \log\hat\sigma_g^2 = \frac{n_g \log s_g^2 + k \log s^2}{n_g + k},

(shrinking the variances themselves would pull a quiet group far towards a
pooled value dominated by the noisy ones),

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
                log_var = len(r) * np.log(np.mean(r**2)) + self.shrinkage * np.log(
                    pooled
                )
                var[lv] = float(np.exp(log_var / (len(r) + self.shrinkage)))
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


def cross_fitted_predictive(
    X: pd.DataFrame,
    y: ArrayLike,
    n_splits: int = 5,
    seed: int = 0,
    shrinkage: float = 10.0,
) -> pd.DataFrame:
    """Out-of-fold mean and error SD from OLS and from the weighted model.

    OLS gets one SD, the root mean squared residual of its training folds;
    the weighted model gets the SD of each member's joining month.
    """
    from sklearn.model_selection import KFold

    y = np.asarray(y, float)
    out = pd.DataFrame(index=range(len(y)), dtype=float)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        Xa, ya, Xb = X.iloc[tr], y[tr], X.iloc[te]
        ols = Pipeline(
            [("prep", make_preprocessor("onehot")), ("model", LinearRegression())]
        ).fit(Xa, ya)
        out.loc[te, "ols_mean"] = ols.predict(Xb)
        out.loc[te, "ols_sd"] = np.sqrt(np.mean((ya - ols.predict(Xa)) ** 2))
        wls = GroupWeightedRegression(shrinkage=shrinkage).fit(Xa, ya)
        out.loc[te, "wls_mean"] = wls.predict(Xb)
        out.loc[te, "wls_sd"] = wls.predict_sd(Xb)
    out["y"] = y
    return out


def predictive_comparison(
    table: pd.DataFrame, groups: ArrayLike, alpha: float = 0.1
) -> pd.DataFrame:
    """Coverage, interval score, CRPS, PIT test and month flags per model."""
    from scipy.stats import norm

    from loyalty_spend.evaluation.scoring import (
        calibration_test,
        crps_gaussian,
        interval_summary,
        pit_gaussian,
    )
    from loyalty_spend.uncertainty.audit import coverage_audit
    from loyalty_spend.uncertainty.conformal import Intervals

    y = table["y"].to_numpy()
    z = norm.ppf(1 - alpha / 2)
    rows = []
    for name in ("ols", "wls"):
        mu, sd = table[f"{name}_mean"].to_numpy(), table[f"{name}_sd"].to_numpy()
        iv = Intervals(mu - z * sd, mu + z * sd)
        audit = coverage_audit(
            iv,
            y,
            pd.DataFrame({"joining_month": np.asarray(groups)}),
            ["joining_month"],
        )
        pit = calibration_test(pit_gaussian(y, mu, sd))
        rows.append(
            {
                "model": name,
                **interval_summary(y, iv.lower, iv.upper, alpha),
                "crps": float(crps_gaussian(y, mu, sd).mean()),
                "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
                "pit_chi2": pit["chi2"],
                "months_flagged": int(audit["flagged"].sum()),
            }
        )
    return pd.DataFrame(rows)


def standardized_residuals(
    model: GroupWeightedRegression, X: pd.DataFrame, y: ArrayLike
) -> FloatArray:
    """Residuals divided by the error SD of each member's group."""
    resid = np.asarray(y, float) - model.predict(X)
    return np.asarray(resid / model.predict_sd(X), float)
