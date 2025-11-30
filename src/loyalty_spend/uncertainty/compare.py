"""Side-by-side evaluation of the interval methods on the same data."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.evaluation.scoring import interval_summary
from loyalty_spend.models.registry import build_model
from loyalty_spend.uncertainty.conformal import (
    ConformalizedQuantileRegressor,
    Intervals,
    SplitConformalRegressor,
)
from loyalty_spend.uncertainty.mondrian import MondrianConformalRegressor
from loyalty_spend.uncertainty.normalized import NormalizedConformalRegressor

Method = Callable[[pd.DataFrame, ArrayLike, pd.DataFrame], Intervals]


def interval_methods(
    alpha: float = 0.1, calibration_size: float = 0.25, seed: int = 0
) -> dict[str, Method]:
    """Each method fits on the training rows and returns intervals for new rows."""

    def split(X: pd.DataFrame, y: ArrayLike, X_new: pd.DataFrame) -> Intervals:
        m = SplitConformalRegressor(build_model("ols"), alpha, calibration_size, seed)
        return m.fit(X, y).predict_interval(X_new)

    def mondrian(X: pd.DataFrame, y: ArrayLike, X_new: pd.DataFrame) -> Intervals:
        m = MondrianConformalRegressor(
            build_model("ols"),
            alpha=alpha,
            calibration_size=calibration_size,
            seed=seed,
        )
        return m.fit(X, y).predict_interval(X_new)

    def normalized(X: pd.DataFrame, y: ArrayLike, X_new: pd.DataFrame) -> Intervals:
        m = NormalizedConformalRegressor(
            build_model("ols"),
            alpha=alpha,
            calibration_size=calibration_size,
            seed=seed,
        )
        return m.fit(X, y).predict_interval(X_new)

    def cqr(X: pd.DataFrame, y: ArrayLike, X_new: pd.DataFrame) -> Intervals:
        m = ConformalizedQuantileRegressor(alpha, calibration_size, seed)
        return m.fit(X, y).predict_interval(X_new)

    def weighted(X: pd.DataFrame, y: ArrayLike, X_new: pd.DataFrame) -> Intervals:
        from scipy.stats import norm

        from loyalty_spend.models.weighted import GroupWeightedRegression

        m = GroupWeightedRegression().fit(X, y)
        half = norm.ppf(1 - alpha / 2) * m.predict_sd(X_new)
        pred = m.predict(X_new)
        return Intervals(pred - half, pred + half)

    return {
        "split conformal": split,
        "Mondrian (prediction bins)": mondrian,
        "normalized conformal": normalized,
        "CQR": cqr,
        "weighted least squares by month": weighted,
    }


def compare_intervals(
    X: pd.DataFrame,
    y: ArrayLike,
    X_new: pd.DataFrame,
    y_new: ArrayLike,
    alpha: float = 0.1,
    calibration_size: float = 0.25,
    seed: int = 0,
    n_groups: int = 5,
) -> pd.DataFrame:
    """Coverage, width, interval score and the worst group coverage per method.

    Groups are quantiles of the OLS forecast for the new rows, the same for
    every method, so the conditional coverage columns are comparable;
    coverage is also reported for the worst and best joining month.
    """
    y_new = np.asarray(y_new, float)
    point = build_model("ols").fit(X, y).predict(X_new)
    groups = pd.qcut(point, n_groups, labels=False, duplicates="drop")
    months = X_new["joining_month"].astype(str).to_numpy()
    rows = []
    for name, method in interval_methods(alpha, calibration_size, seed).items():
        iv = method(X, y, X_new)
        covered = (y_new >= iv.lower) & (y_new <= iv.upper)
        by_group = pd.Series(covered).groupby(groups).mean()
        by_month = pd.Series(covered).groupby(months).mean()
        rows.append(
            {
                "method": name,
                **interval_summary(y_new, iv.lower, iv.upper, alpha),
                "worst_group_coverage": float(by_group.min()),
                "best_group_coverage": float(by_group.max()),
                "worst_month_coverage": float(by_month.min()),
                "best_month_coverage": float(by_month.max()),
            }
        )
    return pd.DataFrame(rows)


def conditional_coverage_comparison(
    X: pd.DataFrame,
    y: ArrayLike,
    columns: tuple[str, ...] = ("loyalty_years", "region", "joining_month"),
    alpha: float = 0.1,
    n_splits: int = 5,
    bandwidth: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fitted intervals for every member, audited by subgroup.

    Each method is fitted on all folds but one and produces intervals for
    the held-out fold, so every member gets an interval from a model that
    did not see them. Returns a summary per method and the full audit.
    """
    from sklearn.model_selection import KFold

    from loyalty_spend.uncertainty.audit import coverage_audit
    from loyalty_spend.uncertainty.localized import LocalizedConformalRegressor

    y = np.asarray(y, float)
    methods = dict(interval_methods(alpha, seed=seed))
    for col in ("loyalty_years", "joining_month"):

        def mondrian(
            Xa: pd.DataFrame, ya: ArrayLike, Xb: pd.DataFrame, col: str = col
        ) -> Intervals:
            # Months have about 17 calibration members each; 15 keeps the
            # group quantile finite without pooling every month.
            m = MondrianConformalRegressor(
                build_model("ols"), by=col, alpha=alpha, min_group_size=15, seed=seed
            )
            return m.fit(Xa, ya).predict_interval(Xb)

        methods[f"Mondrian ({col})"] = mondrian

    def localized(Xa: pd.DataFrame, ya: ArrayLike, Xb: pd.DataFrame) -> Intervals:
        m = LocalizedConformalRegressor(build_model("ols"), alpha, bandwidth, seed=seed)
        return m.fit(Xa, ya).predict_interval(Xb)

    methods[f"localised (h = {bandwidth:g})"] = localized
    summaries, audits = [], []
    for name, method in methods.items():
        lower, upper = np.empty(len(y)), np.empty(len(y))
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
            iv = method(X.iloc[tr], y[tr], X.iloc[te])
            lower[te], upper[te] = iv.lower, iv.upper
        iv = Intervals(lower, upper)
        audit = coverage_audit(iv, y, X, list(columns), alpha).assign(method=name)
        audits.append(audit)
        worst = audit.loc[audit["coverage"].idxmin()]
        summaries.append(
            {
                "method": name,
                **interval_summary(y, lower, upper, alpha),
                "levels_flagged": int(audit["flagged"].sum()),
                "worst_level": f"{worst['attribute']}={worst['level']}",
                "worst_coverage": float(worst["coverage"]),
            }
        )
    return pd.DataFrame(summaries), pd.concat(audits, ignore_index=True)
