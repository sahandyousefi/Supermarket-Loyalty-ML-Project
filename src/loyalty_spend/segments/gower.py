r"""Gower distance for mixed numeric, ordinal and nominal features.

The distance between members :math:`i` and :math:`j` averages per-feature
dissimilarities in :math:`[0, 1]` (Gower, 1971):

.. math::

    d_{ij} = \frac{\sum_k w_k\, \delta_{ijk}\, d_{ijk}}{\sum_k w_k\, \delta_{ijk}},

where :math:`d_{ijk} = |x_{ik} - x_{jk}| / R_k` for numeric features with
range :math:`R_k`, the rank difference divided by the number of levels minus
one for ordinal features (Podani, 1999), and :math:`\mathbb 1\{x_{ik} \ne
x_{jk}\}` for nominal features. :math:`\delta_{ijk} = 0` when either value
is missing, so missing values reduce the information about a pair instead
of being imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loyalty_spend.data.schema import LOYALTY_ORDER, NUMERIC

ORDINAL: dict[str, list[str]] = {"loyalty_years": LOYALTY_ORDER}


def _numeric_part(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = ~np.isnan(x)
    span = np.nanmax(x) - np.nanmin(x) if ok.any() else 0.0
    d = (
        np.abs(x[:, None] - x[None, :]) / span
        if span > 0
        else np.zeros((len(x), len(x)))
    )
    valid = ok[:, None] & ok[None, :]
    return np.where(valid, d, 0.0), valid


def gower_matrix(
    X: pd.DataFrame,
    numeric: list[str] | None = None,
    ordinal: dict[str, list[str]] | None = None,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Symmetric matrix of Gower distances between the rows of ``X``.

    Columns listed in ``numeric`` are treated as interval scaled, columns in
    ``ordinal`` by their rank among the given levels, and every other column
    as nominal.
    """
    numeric = [c for c in (NUMERIC if numeric is None else numeric) if c in X]
    ordinal = {
        k: v for k, v in (ORDINAL if ordinal is None else ordinal).items() if k in X
    }
    weights = weights or {}
    n = len(X)
    total = np.zeros((n, n))
    norm = np.zeros((n, n))
    for col in X.columns:
        w = float(weights.get(col, 1.0))
        if w == 0:
            continue
        if col in numeric:
            d, valid = _numeric_part(
                pd.to_numeric(X[col], errors="coerce").to_numpy(float)
            )
        elif col in ordinal:
            levels = ordinal[col]
            codes = pd.Categorical(X[col].astype("string"), categories=levels).codes
            ranks = np.where(codes >= 0, codes / max(len(levels) - 1, 1), np.nan)
            d, valid = _numeric_part(ranks.astype(float))
            d = np.where(valid, np.abs(ranks[:, None] - ranks[None, :]), 0.0)
        else:
            codes = pd.factorize(X[col].astype("string"))[0]
            valid = (codes[:, None] >= 0) & (codes[None, :] >= 0)
            d = np.where(valid, codes[:, None] != codes[None, :], 0.0).astype(float)
        total += w * d
        norm += w * valid
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(norm > 0, total / norm, np.nan)
    np.fill_diagonal(out, 0.0)
    return out
