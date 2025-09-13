"""Stability of feature importance across cross-validation folds.

An importance ranking that changes from fold to fold should not be
reported as a finding. Grouped permutation importance is computed on the
held-out part of every fold, and agreement between folds is summarised by
Kendall's coefficient of concordance W, which is 1 when every fold ranks
the features identically and near 0 when the rankings are unrelated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, clone

from loyalty_spend.explain.attribution import grouped_permutation_importance


def kendalls_w(ranks: np.ndarray) -> float:
    """Concordance of ``m`` rankings (rows) of ``n`` items (columns)."""
    m, n = ranks.shape
    totals = ranks.sum(axis=0)
    s = float(((totals - totals.mean()) ** 2).sum())
    return 12 * s / (m**2 * (n**3 - n))


def importance_stability(
    model: BaseEstimator | Callable[[], BaseEstimator],
    X: pd.DataFrame,
    y: ArrayLike,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_repeats: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Importance per fold and feature, and Kendall's W across folds."""
    y = np.asarray(y, float)
    frames = []
    for k, (tr, te) in enumerate(splits):
        est = (
            model()
            if callable(model) and not isinstance(model, BaseEstimator)
            else clone(model)
        )
        est.fit(X.iloc[tr], y[tr])
        imp = grouped_permutation_importance(est, X.iloc[te], y[te], n_repeats, seed)
        frames.append(imp.assign(fold=k))
    table = pd.concat(frames, ignore_index=True)
    wide = table.pivot(index="fold", columns="feature", values="rmse_increase")
    ranks = np.vstack([rankdata(-row) for row in wide.to_numpy()])
    return table, kendalls_w(ranks)
