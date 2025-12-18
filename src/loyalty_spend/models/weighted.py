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

from loyalty_spend.arrays import FloatArray, ObjectArray, Split
from loyalty_spend.features.preprocess import make_preprocessor
from loyalty_spend.models.registry import register


class CohortScale:
    r"""Error SD of a joining cohort from the spread of its first-month spend.

    Across joining months the residual SD and the within-month SD of
    first-month spend move together. The fit is a line on the log scale,
    :math:`\log\sigma_g = a + b \log s_g`, with :math:`s_g` the SD of
    first-month spend in month :math:`g`; a cohort that has spent its first
    month but not yet a year gets :math:`\exp(a + b\log s)` as its error SD.
    """

    def fit(
        self, months: ArrayLike, first_month: ArrayLike, residuals: ArrayLike
    ) -> CohortScale:
        frame = pd.DataFrame(
            {
                "g": np.asarray(months).astype(str),
                "x": np.asarray(first_month, float),
                "e": np.asarray(residuals, float),
            }
        ).dropna()
        stats_ = frame.groupby("g").agg(
            spread=("x", "std"), sigma=("e", lambda e: float(np.sqrt(np.mean(e**2))))
        )
        u, v = np.log(stats_["spread"].to_numpy()), np.log(stats_["sigma"].to_numpy())
        self.slope_, self.intercept_ = (float(c) for c in np.polyfit(u, v, 1))
        fitted = self.intercept_ + self.slope_ * u
        self.r2_ = float(1 - np.sum((v - fitted) ** 2) / np.sum((v - v.mean()) ** 2))
        self.table_ = stats_.assign(fitted=np.exp(fitted)).reset_index()
        return self

    def predict(self, first_month: ArrayLike) -> float:
        """Error SD of a cohort with these first-month amounts."""
        spread = float(np.nanstd(np.asarray(first_month, float), ddof=1))
        return float(np.exp(self.intercept_ + self.slope_ * np.log(spread)))


class GroupWeightedRegression(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        group: str = "joining_month",
        shrinkage: float = 10.0,
        n_iter: int = 3,
        estimator: BaseEstimator | None = None,
        spread_column: str | None = "first_month",
        min_cohort: int = 10,
    ) -> None:
        self.group = group
        self.shrinkage = shrinkage
        self.n_iter = n_iter
        self.estimator = estimator
        self.spread_column = spread_column
        self.min_cohort = min_cohort

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
        self.cohort_: CohortScale | None = None
        if self.spread_column is not None and len(var) >= 3:
            self.cohort_ = CohortScale().fit(labels, X[self.spread_column], resid)
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        return np.asarray(self.model_.predict(X), float)

    def predict_sd(self, X: pd.DataFrame) -> FloatArray:
        """Error standard deviation of each member's group.

        A group not seen in training gets the SD that the cohort model
        predicts from the spread of its members' first-month spend in ``X``,
        when at least ``min_cohort`` of them are present; otherwise it gets
        the pooled SD.
        """
        labels = self._labels(X)
        sd = {lv: float(np.sqrt(v)) for lv, v in self.variance_.items()}
        for lv in set(labels) - set(sd):
            sd[lv] = float(np.sqrt(self.pooled_variance_))
            if self.cohort_ is not None and self.spread_column is not None:
                values = X.loc[labels == lv, self.spread_column].dropna()
                if len(values) >= self.min_cohort:
                    sd[lv] = self.cohort_.predict(values)
        return np.array([sd[lv] for lv in labels])


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


def cross_fitted_scaled_residuals(
    name: str, X: pd.DataFrame, y: ArrayLike, splits: list[Split]
) -> FloatArray:
    """Out-of-fold residuals divided by the out-of-fold error SD of each member.

    Their absolute values are the scores of normalized split conformal
    prediction: the ``1 - alpha`` quantile times a member's SD gives an
    interval half-width that adapts to the member's joining month.
    """
    from loyalty_spend.models.registry import build_model

    y = np.asarray(y, float)
    out = np.empty(len(y))
    for tr, te in splits:
        model = build_model(name).fit(X.iloc[tr], y[tr])
        out[te] = (y[te] - model.predict(X.iloc[te])) / model.predict_sd(X.iloc[te])
    return out


def leave_one_month_out(
    X: pd.DataFrame,
    y: ArrayLike,
    alpha: float = 0.1,
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """How well the cohort model predicts the error scale of a held-out month.

    Out-of-fold forecasts come from the weighted model with every month in
    training, so the point forecast of each member is the same in every
    row. Only the error scale is held out: the cohort model is fitted on
    the other eleven months and predicts the held-out month's SD from its
    first-month spread. Normal intervals with that SD are compared with
    intervals that use the pooled SD of the other months.
    """
    from scipy.stats import norm
    from sklearn.model_selection import KFold

    from loyalty_spend.evaluation.scoring import interval_score

    y = np.asarray(y, float)
    oof = np.empty(len(y))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        oof[te] = GroupWeightedRegression().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    resid = y - oof
    months = X["joining_month"].astype(str).to_numpy()
    z = norm.ppf(1 - alpha / 2)
    rows = []
    for g in np.unique(months):
        held = months == g
        if held.sum() < 10:
            continue
        scale = CohortScale().fit(months[~held], X["first_month"][~held], resid[~held])
        predicted = scale.predict(X.loc[held, "first_month"])
        pooled = float(np.sqrt(np.mean(resid[~held] ** 2)))
        actual = float(np.sqrt(np.mean(resid[held] ** 2)))
        row = {"month": g, "n": int(held.sum()), "actual_sd": actual}
        for name, sd in (("cohort", predicted), ("pooled", pooled)):
            lo, hi = oof[held] - z * sd, oof[held] + z * sd
            row[f"{name}_sd"] = sd
            row[f"{name}_coverage"] = float(np.mean((y[held] >= lo) & (y[held] <= hi)))
            row[f"{name}_score"] = float(interval_score(y[held], lo, hi, alpha).mean())
        rows.append(row)
    return pd.DataFrame(rows)
