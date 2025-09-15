"""Learning curves with intervals.

How much would more members improve the forecast? The model is refitted on
growing random subsets of the training folds and scored on the held-out
folds; repeating over folds gives an interval for every training size. A
curve that has flattened says that more data of the same kind will not
help, and effort belongs to new attributes instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone


def learning_curve(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    splits: list[tuple[np.ndarray, np.ndarray]],
    sizes: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 0,
) -> pd.DataFrame:
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    rows = []
    for k, (tr, te) in enumerate(splits):
        order = rng.permutation(tr)
        for share in sizes:
            n = max(20, int(round(share * len(tr))))
            idx = order[:n]
            est = clone(model).fit(X.iloc[idx], y[idx])
            rmse = float(np.sqrt(np.mean((est.predict(X.iloc[te]) - y[te]) ** 2)))
            rows.append({"fold": k, "n_train": n, "rmse": rmse})
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("n_train")["rmse"].agg(["mean", "std", "count"]).reset_index()
    )
    summary["lower"] = summary["mean"] - 1.96 * summary["std"] / np.sqrt(
        summary["count"]
    )
    summary["upper"] = summary["mean"] + 1.96 * summary["std"] / np.sqrt(
        summary["count"]
    )
    return summary.rename(columns={"mean": "rmse"}).drop(columns="count")


def has_plateaued(curve: pd.DataFrame, tolerance: float = 0.02) -> bool:
    """Whether the last step improved RMSE by less than ``tolerance`` (relative)."""
    last, previous = curve["rmse"].iloc[-1], curve["rmse"].iloc[-2]
    return bool((previous - last) / previous < tolerance)
