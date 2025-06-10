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

    return {
        "split conformal": split,
        "Mondrian (prediction bins)": mondrian,
        "normalized conformal": normalized,
        "CQR": cqr,
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
    every method, so the conditional coverage columns are comparable.
    """
    y_new = np.asarray(y_new, float)
    point = build_model("ols").fit(X, y).predict(X_new)
    groups = pd.qcut(point, n_groups, labels=False, duplicates="drop")
    rows = []
    for name, method in interval_methods(alpha, calibration_size, seed).items():
        iv = method(X, y, X_new)
        covered = (y_new >= iv.lower) & (y_new <= iv.upper)
        by_group = pd.Series(covered).groupby(groups).mean()
        rows.append(
            {
                "method": name,
                **interval_summary(y_new, iv.lower, iv.upper, alpha),
                "worst_group_coverage": float(by_group.min()),
                "best_group_coverage": float(by_group.max()),
            }
        )
    return pd.DataFrame(rows)
