r"""Variance reduction for the experiment with pre-treatment covariates.

CUPED (Deng et al., 2013) replaces spend :math:`Y` by
:math:`Y - \theta(X - \bar X)` with :math:`\theta = \mathrm{cov}(Y, X) /
\mathrm{var}(X)` estimated on both arms together; with one covariate the
variance of the effect estimate falls by the factor :math:`1 - \rho^2`.
Regression adjustment with treatment-by-covariate interactions (Lin, 2013)
generalises this to several covariates, cannot hurt asymptotic precision
under randomisation, and with HC2 standard errors is valid without
assuming the model is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.linalg import qr


@dataclass
class EffectEstimate:
    estimate: float
    se: float
    method: str

    @property
    def interval(self) -> tuple[float, float]:
        return self.estimate - 1.96 * self.se, self.estimate + 1.96 * self.se


def difference_in_means(y: ArrayLike, treat: ArrayLike) -> EffectEstimate:
    y, t = np.asarray(y, float), np.asarray(treat, bool)
    a, b = y[t], y[~t]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return EffectEstimate(float(a.mean() - b.mean()), float(se), "difference")


def cuped(y: ArrayLike, x: ArrayLike, treat: ArrayLike) -> EffectEstimate:
    y, x = np.asarray(y, float), np.asarray(x, float)
    cov = np.cov(y, x, ddof=1)
    theta = cov[0, 1] / cov[1, 1]
    adjusted = y - theta * (x - x.mean())
    est = difference_in_means(adjusted, treat)
    return EffectEstimate(est.estimate, est.se, "cuped")


def _full_rank_columns(D: np.ndarray, keep_first: int = 2) -> np.ndarray:
    """Indices of a set of linearly independent columns, keeping the first ones."""
    _, r, piv = qr(D[:, keep_first:], mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int((diag > diag[0] * 1e-10).sum()) if diag.size else 0
    return np.r_[np.arange(keep_first), keep_first + np.sort(piv[:rank])]


def lin_adjusted(
    y: ArrayLike,
    X: pd.DataFrame,
    treat: ArrayLike,
    min_rows_per_column: int = 10,
) -> EffectEstimate:
    """OLS of spend on treatment, centred covariates and their interactions.

    Interactions double the number of columns; when there are fewer than
    ``min_rows_per_column`` members per column they are left out and the
    estimator becomes ordinary covariance adjustment, which is still
    consistent under randomisation. Collinear columns, such as dummies of
    levels absent from a small sample, are dropped.
    """
    y = np.asarray(y, float)
    t = np.asarray(treat, float)
    Z = pd.get_dummies(X, drop_first=True, dtype=float).to_numpy()
    Z = Z - Z.mean(axis=0)
    interact = len(y) >= min_rows_per_column * (2 + 2 * Z.shape[1])
    blocks = [np.ones(len(y)), t, Z] + ([Z * t[:, None]] if interact else [])
    D = np.column_stack(blocks)
    D = D[:, _full_rank_columns(D)]
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    bread = np.linalg.inv(D.T @ D)
    hat = np.einsum("ij,jk,ik->i", D, bread, D)
    meat = D.T @ (D * (resid**2 / (1 - np.minimum(hat, 0.999)))[:, None])
    cov = bread @ meat @ bread
    method = "lin" if interact else "ancova"
    return EffectEstimate(float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0))), method)
