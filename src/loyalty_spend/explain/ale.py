r"""Accumulated local effects (Apley and Zhu, 2020).

Partial dependence averages predictions over the marginal distribution of
the other features, so with correlated features it evaluates the model at
combinations that never occur. ALE avoids this: within each quantile bin
:math:`(z_{k-1}, z_k]` of feature :math:`j` it averages the change of the
prediction between the bin edges over the members actually in the bin, and
accumulates these local changes:

.. math::

    \hat f_j(x) = \sum_{k \le k(x)} \frac{1}{n_k} \sum_{i \in k}
    \big[f(z_k, x_{i,-j}) - f(z_{k-1}, x_{i,-j})\big] - c,

with :math:`c` chosen so the effect averages to zero over the data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Predict = Callable[[pd.DataFrame], np.ndarray]


def ale_1d(
    predict: Predict, X: pd.DataFrame, feature: str, n_bins: int = 20
) -> pd.DataFrame:
    """ALE curve of a numeric feature at the bin edges."""
    data = X[X[feature].notna()].reset_index(drop=True)
    values = data[feature].astype(float).to_numpy()
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        raise ValueError(f"'{feature}' has a single value")
    bins = np.clip(np.searchsorted(edges, values, side="left"), 1, len(edges) - 1)
    effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for k in range(1, len(edges)):
        members = data[bins == k]
        if members.empty:
            continue
        upper = predict(members.assign(**{feature: edges[k]}))
        lower = predict(members.assign(**{feature: edges[k - 1]}))
        effects[k - 1] = float(np.mean(upper - lower))
        counts[k - 1] = len(members)
    accumulated = np.r_[0.0, np.cumsum(effects)]
    # Centre: the mean effect over members, using the bin midpoint values.
    mid = (accumulated[:-1] + accumulated[1:]) / 2
    centre = float(np.sum(mid * counts) / counts.sum())
    return pd.DataFrame(
        {
            "feature": feature,
            "value": edges,
            "effect": accumulated - centre,
            "n_in_bin": np.r_[0, counts],
        }
    )
