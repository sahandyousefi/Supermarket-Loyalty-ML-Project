r"""Covariate balance between treated and control units.

The standardized mean difference of a covariate column :math:`x` is

.. math::

    d = \frac{\bar x_1^{w} - \bar x_0^{w}}{\sqrt{(s_1^2 + s_0^2)/2}},

with weighted group means and the *unweighted* group variances in the
denominator, so that weighting can only change the numerator and the
before and after values are comparable (Austin, 2009). Categorical
covariates are expanded into one indicator per level. An absolute value
below 0.1 is the usual threshold for adequate balance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def indicator_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns as they are, categorical columns as level indicators."""
    parts = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            parts.append(X[[col]].astype(float))
        else:
            dummies = pd.get_dummies(X[col].astype(str), prefix=col, prefix_sep="=")
            parts.append(dummies.astype(float))
    return pd.concat(parts, axis=1)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w[:, None] * x).sum(axis=0) / w.sum()


def standardized_mean_differences(
    X: pd.DataFrame, t: ArrayLike, weights: ArrayLike | None = None
) -> pd.Series:
    """SMD of every indicator or numeric column, optionally after weighting."""
    t = np.asarray(t, int)
    D = indicator_frame(X)
    x = D.to_numpy()
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float)
    m1 = _weighted_mean(x[t == 1], w[t == 1])
    m0 = _weighted_mean(x[t == 0], w[t == 0])
    sd = np.sqrt((x[t == 1].var(axis=0, ddof=1) + x[t == 0].var(axis=0, ddof=1)) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        smd = np.where(sd > 0, (m1 - m0) / sd, 0.0)
    return pd.Series(smd, index=D.columns)


def ate_weights(propensity: ArrayLike, t: ArrayLike) -> np.ndarray:
    """Inverse probability weights for the average treatment effect."""
    e = np.asarray(propensity, float)
    t = np.asarray(t, int)
    return np.where(t == 1, 1 / e, 1 / (1 - e))


def att_weights(propensity: ArrayLike, t: ArrayLike) -> np.ndarray:
    """Weights that reweight controls to the treated population."""
    e = np.asarray(propensity, float)
    t = np.asarray(t, int)
    return np.where(t == 1, 1.0, e / (1 - e))


def balance_table(
    X: pd.DataFrame, t: ArrayLike, propensity: ArrayLike, threshold: float = 0.1
) -> pd.DataFrame:
    """SMD before and after ATE weighting, with a balance flag."""
    before = standardized_mean_differences(X, t)
    after = standardized_mean_differences(X, t, ate_weights(propensity, t))
    table = pd.DataFrame({"smd_unweighted": before, "smd_weighted": after})
    table["balanced"] = table["smd_weighted"].abs() < threshold
    return table.rename_axis("covariate").reset_index()
