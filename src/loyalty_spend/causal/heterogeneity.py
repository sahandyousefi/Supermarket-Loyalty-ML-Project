r"""Inference on effect heterogeneity from any CATE model.

Machine-learned CATE predictions have no standard errors. Chernozhukov et
al. (2018, "generic machine learning inference") treat them as a proxy
:math:`S(x)` and test what the proxy reveals. With doubly robust scores
:math:`\psi_i`, the best linear predictor (BLP) of the CATE given the proxy is
estimated by

.. math::

    \psi_i = \beta_1 + \beta_2 \big(S(x_i) - \bar S\big) + \epsilon_i .

:math:`\beta_1` estimates the ATE; :math:`\beta_2 = 1` means the proxy is a
perfect predictor and :math:`\beta_2 = 0` that it carries no information
about heterogeneity. Rejecting :math:`\beta_2 = 0` is evidence that the
effect varies and that the model found how. The proxy must be predicted out
of fold, so that its errors are independent of the scores it is tested on.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.model_selection import StratifiedKFold


def crossfit_cate(
    factory: Callable[[], object],
    X: pd.DataFrame,
    t: ArrayLike,
    y: ArrayLike,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold CATE predictions from a meta-learner factory."""
    t = np.asarray(t, int)
    y = np.asarray(y, float)
    out = np.empty(len(y))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in folds.split(X, t):
        model = factory().fit(X.iloc[tr], t[tr], y[tr])  # type: ignore[attr-defined]
        out[te] = model.effect(X.iloc[te])
    return out


def _ols_hc1(Z: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    resid = v - Z @ beta
    n, k = Z.shape
    bread = np.linalg.pinv(Z.T @ Z)
    meat = Z.T @ (Z * resid[:, None] ** 2)
    cov = bread @ meat @ bread * n / (n - k)
    return beta, np.sqrt(np.diag(cov))


def blp_test(proxy: ArrayLike, scores: ArrayLike) -> dict[str, float]:
    """Best linear predictor of the CATE with heteroscedasticity-robust errors."""
    s = np.asarray(proxy, float)
    psi = np.asarray(scores, float)
    Z = np.column_stack([np.ones_like(s), s - s.mean()])
    beta, se = _ols_hc1(Z, psi)
    return {
        "ate": float(beta[0]),
        "ate_se": float(se[0]),
        "heterogeneity": float(beta[1]),
        "heterogeneity_se": float(se[1]),
        "heterogeneity_p_value": float(2 * stats.norm.sf(abs(beta[1] / se[1]))),
    }
