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


def gates(proxy: ArrayLike, scores: ArrayLike, n_groups: int = 5) -> dict[str, object]:
    """Sorted group average treatment effects and the top-bottom difference.

    Members are split into ``n_groups`` groups by the proxy; group 1 has the
    largest predicted effects. The estimate in each group is the mean of the
    doubly robust scores, and the difference between the first and last
    group has a normal interval from the two independent group means.
    """
    from loyalty_spend.causal.uplift import effect_by_group

    table = effect_by_group(proxy, scores, n_groups)
    top, bottom = table.iloc[0], table.iloc[-1]
    diff = float(top["estimated"] - bottom["estimated"])
    se = float(np.hypot(top["se"], bottom["se"]))
    return {
        "groups": table,
        "top_minus_bottom": diff,
        "top_minus_bottom_se": se,
        "p_value": float(2 * stats.norm.sf(abs(diff / se))) if se > 0 else float("nan"),
    }


def clan(X: pd.DataFrame, proxy: ArrayLike, n_groups: int = 5) -> pd.DataFrame:
    """Covariate profile of the most and least affected groups.

    The classification analysis of Chernozhukov et al. (2018) compares the
    average characteristics of the top and bottom groups, which tells the
    business who responds to the promotion.
    """
    from loyalty_spend.causal.balance import indicator_frame

    s = np.asarray(proxy, float)
    ranks = pd.Series(-s).rank(method="first").to_numpy()
    groups = np.ceil(ranks * n_groups / len(s)).astype(int)
    D = indicator_frame(X.reset_index(drop=True))
    top = D[groups == 1].mean()
    bottom = D[groups == n_groups].mean()
    out = pd.DataFrame({"most_affected": top, "least_affected": bottom})
    out["difference"] = out["most_affected"] - out["least_affected"]
    return (
        out.reindex(out["difference"].abs().sort_values(ascending=False).index)
        .rename_axis("covariate")
        .reset_index()
    )
