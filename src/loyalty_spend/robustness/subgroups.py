"""Forecast errors by subgroup with bootstrap intervals.

An overall RMSE can hide a group that is forecast badly. Errors are computed
from out-of-fold predictions, so every member is scored by a model that did
not see them, and every subgroup gets a percentile bootstrap interval for
its RMSE and its mean error (bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.arrays import FloatArray


def _bootstrap(
    errors: FloatArray, n_boot: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    idx = rng.integers(0, len(errors), size=(n_boot, len(errors)))
    sample = errors[idx]
    return np.sqrt((sample**2).mean(axis=1)), sample.mean(axis=1)


def subgroup_errors(
    y: ArrayLike,
    pred: ArrayLike,
    frame: pd.DataFrame,
    columns: list[str],
    n_boot: int = 1000,
    min_size: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    err = np.asarray(pred, float) - np.asarray(y, float)
    rows = []
    for col in columns:
        labels = frame[col].astype("string").fillna("<missing>").to_numpy()
        for level in sorted(set(labels)):
            e = err[labels == level]
            if len(e) < min_size:
                continue
            rmse_b, bias_b = _bootstrap(e, n_boot, rng)
            rows.append(
                {
                    "attribute": col,
                    "level": level,
                    "n": len(e),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "rmse_lower": float(np.quantile(rmse_b, 0.025)),
                    "rmse_upper": float(np.quantile(rmse_b, 0.975)),
                    "bias": float(e.mean()),
                    "bias_lower": float(np.quantile(bias_b, 0.025)),
                    "bias_upper": float(np.quantile(bias_b, 0.975)),
                }
            )
    table = pd.DataFrame(rows)
    overall = float(np.sqrt(np.mean(err**2)))
    table["rmse_ratio"] = table["rmse"] / overall
    return table
