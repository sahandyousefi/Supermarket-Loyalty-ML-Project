r"""Influence diagnostics for the linear champion.

For the design matrix :math:`A` of the fitted pipeline (with intercept),
the hat values are :math:`h_i = a_i^\top (A^\top A)^{-1} a_i`. With
internally studentized residuals :math:`r_i = e_i / (s\sqrt{1 - h_i})`,
Cook's distance is :math:`D_i = r_i^2 h_i / (p(1 - h_i))` and DFBETAS
measure how much each coefficient moves, in standard errors, when member
:math:`i` is left out (Belsley, Kuh and Welsch, 1980). The usual flags are
:math:`h_i > 2p/n`, :math:`D_i > 4/n` and :math:`|\mathrm{DFBETAS}| > 2/\sqrt n`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline


def influence_table(pipeline: Pipeline, X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
    y = np.asarray(y, float)
    prep = pipeline[:-1]
    A = np.column_stack([np.ones(len(X)), np.asarray(prep.transform(X), float)])
    n, p = A.shape
    inv = np.linalg.pinv(A.T @ A)
    beta = inv @ A.T @ y
    e = y - A @ beta
    h = np.einsum("ij,jk,ik->i", A, inv, A)
    p_eff = int(round(h.sum()))
    s2 = e @ e / (n - p_eff)
    r = e / np.sqrt(s2 * (1 - h))
    cooks = r**2 * h / (p_eff * (1 - h))
    # Leave-one-out residual variance for the externally studentized DFBETAS.
    s2_loo = ((n - p_eff) * s2 - e**2 / (1 - h)) / (n - p_eff - 1)
    delta = (inv @ A.T).T * (e / (1 - h))[:, None]
    dfbetas = delta / np.sqrt(np.outer(s2_loo, np.diag(inv)))
    out = pd.DataFrame(
        {
            "hat": h,
            "studentized_residual": r,
            "cooks_distance": cooks,
            "max_abs_dfbetas": np.abs(dfbetas[:, 1:]).max(axis=1),
        },
        index=X.index,
    )
    out["flag_hat"] = out["hat"] > 2 * p_eff / n
    out["flag_cooks"] = out["cooks_distance"] > 4 / n
    out["flag_dfbetas"] = out["max_abs_dfbetas"] > 2 / np.sqrt(n)
    return out
