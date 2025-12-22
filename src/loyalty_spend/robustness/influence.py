r"""Influence diagnostics for the linear champion.

For the design matrix :math:`A` of the fitted pipeline (with intercept),
the hat values are :math:`h_i = a_i^\top (A^\top A)^{-1} a_i`. With
internally studentized residuals :math:`r_i = e_i / (s\sqrt{1 - h_i})`,
Cook's distance is :math:`D_i = r_i^2 h_i / (p(1 - h_i))` and DFBETAS
measure how much each coefficient moves, in standard errors, when member
:math:`i` is left out (Belsley, Kuh and Welsch, 1980). The usual flags are
:math:`h_i > 2p/n`, :math:`D_i > 4/n` and :math:`|\mathrm{DFBETAS}| > 2/\sqrt n`.

A member who is the only one with some category level has :math:`h_i = 1`:
the fit passes through them, their residual is zero and leaving them out
removes the coefficient of that level, so the studentized residual, Cook's
distance and DFBETAS are undefined. They are reported as missing with
``hat_one`` set, rather than as the result of a division by zero.
Coefficients of levels absent from the data are left out of DFBETAS.
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
    one = h > 1 - 1e-10
    q = np.where(one, np.nan, 1 - h)
    r = e / np.sqrt(s2 * q)
    cooks = r**2 * h / (p_eff * q)
    # Leave-one-out residual variance for the externally studentized DFBETAS.
    s2_loo = ((n - p_eff) * s2 - e**2 / q) / (n - p_eff - 1)
    delta = (inv @ A.T).T * (e / q)[:, None]
    diag = np.diag(inv)
    present = diag[1:] > 1e-12
    scale = np.sqrt(np.outer(s2_loo, diag[1:][present]))
    dfbetas = delta[:, 1:][:, present] / scale
    max_dfbetas = np.where(one, np.nan, np.abs(np.nan_to_num(dfbetas)).max(axis=1))
    out = pd.DataFrame(
        {
            "hat": h,
            "studentized_residual": r,
            "cooks_distance": cooks,
            "max_abs_dfbetas": max_dfbetas,
            "hat_one": one,
        },
        index=X.index,
    )
    out["flag_hat"] = out["hat"] > 2 * p_eff / n
    out["flag_cooks"] = out["cooks_distance"] > 4 / n
    out["flag_dfbetas"] = out["max_abs_dfbetas"] > 2 / np.sqrt(n)
    return out
