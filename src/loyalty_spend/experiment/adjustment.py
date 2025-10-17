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


def lin_adjusted(y: ArrayLike, X: pd.DataFrame, treat: ArrayLike) -> EffectEstimate:
    """OLS of spend on treatment, centred covariates and their interactions."""
    y = np.asarray(y, float)
    t = np.asarray(treat, float)
    Z = pd.get_dummies(X, drop_first=True, dtype=float).to_numpy()
    Z = Z - Z.mean(axis=0)
    D = np.column_stack([np.ones(len(y)), t, Z, Z * t[:, None]])
    # Drop columns that are constant after centring (empty dummy levels).
    keep = np.r_[True, True, np.abs(D[:, 2:]).sum(axis=0) > 0]
    D = D[:, keep]
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    bread = np.linalg.pinv(D.T @ D)
    hat = np.einsum("ij,jk,ik->i", D, bread, D)
    meat = D.T @ (D * (resid**2 / (1 - np.minimum(hat, 0.999)))[:, None])
    cov = bread @ meat @ bread
    return EffectEstimate(float(beta[1]), float(np.sqrt(cov[1, 1])), "lin")
