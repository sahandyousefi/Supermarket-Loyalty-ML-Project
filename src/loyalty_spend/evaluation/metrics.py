"""Regression metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error; undefined for zero targets."""
    y = np.asarray(y_true, float)
    if np.any(y == 0):
        raise ValueError("MAPE is undefined when a target value is zero")
    return float(np.mean(np.abs((y - np.asarray(y_pred, float)) / y)))


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }
