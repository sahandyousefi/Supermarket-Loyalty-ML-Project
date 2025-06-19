r"""Sensitivity of the regression estimate to an unobserved confounder.

Following Cinelli and Hazlett (2020), the treatment coefficient of the OLS
regression :math:`Y \sim T + X` is compared with the coefficient that would
be obtained if an omitted confounder :math:`Z` were added. The confounder is
described by two partial :math:`R^2` values: the share of the residual
variance of :math:`T` it explains given :math:`X`
(:math:`R^2_{T \sim Z \mid X}`) and the share of the residual variance of
:math:`Y` it explains given :math:`T` and :math:`X`
(:math:`R^2_{Y \sim Z \mid T, X}`). The absolute bias is then

.. math::

    |\widehat{\mathrm{bias}}| = \mathrm{se}(\hat\tau)\,\sqrt{\mathrm{df}}\,
    \sqrt{\frac{R^2_{Y\sim Z\mid T,X}\,R^2_{T\sim Z\mid X}}{1 - R^2_{T\sim Z\mid X}}}.

The robustness value :math:`\mathrm{RV}_q` is the strength, equal on both
partial :math:`R^2`, that a confounder needs to reduce the estimate by
:math:`100q\%`; :math:`\mathrm{RV}_{q,\alpha}` is the strength needed to make
the adjusted interval include that value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from loyalty_spend.causal.balance import indicator_frame
from loyalty_spend.causal.data import CausalData


@dataclass(frozen=True)
class RegressionFit:
    estimate: float
    se: float
    df: int

    @property
    def t_stat(self) -> float:
        return self.estimate / self.se

    @property
    def partial_r2(self) -> float:
        """Share of the residual variance of Y explained by T given X."""
        t2 = self.t_stat**2
        return float(t2 / (t2 + self.df))


def _ols_coefficient(Z: np.ndarray, y: np.ndarray, column: int) -> RegressionFit:
    Z = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    df = len(y) - np.linalg.matrix_rank(Z)
    sigma2 = resid @ resid / df
    cov = sigma2 * np.linalg.pinv(Z.T @ Z)
    j = column + 1
    return RegressionFit(float(beta[j]), float(np.sqrt(cov[j, j])), int(df))


def treatment_regression(data: CausalData) -> RegressionFit:
    """OLS of the outcome on the treatment and the covariate indicators."""
    covariates = indicator_frame(data.X)
    # One level of each categorical covariate is the reference.
    covariates = covariates.loc[:, covariates.std() > 0]
    Z = np.column_stack([data.t, covariates.to_numpy()])
    return _ols_coefficient(Z, data.y, 0)


def bias_bound(fit: RegressionFit, r2_yz: float, r2_tz: float) -> float:
    """Largest absolute bias from a confounder with the given partial R-squared."""
    if not (0 <= r2_yz <= 1 and 0 <= r2_tz < 1):
        raise ValueError("partial R-squared values must be in [0, 1)")
    return float(fit.se * np.sqrt(fit.df) * np.sqrt(r2_yz * r2_tz / (1 - r2_tz)))


def robustness_value(
    fit: RegressionFit, q: float = 1.0, alpha: float | None = None
) -> float:
    """Robustness value ``RV_q`` or, with ``alpha``, ``RV_{q, alpha}``."""
    f = q * abs(fit.t_stat) / np.sqrt(fit.df)
    if alpha is not None:
        f = f - stats.t.ppf(1 - alpha / 2, fit.df - 1) / np.sqrt(fit.df - 1)
        if f <= 0:
            return 0.0
    return float(0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2))


def sensitivity_summary(
    data: CausalData, q: float = 1.0, alpha: float = 0.05
) -> dict[str, float]:
    fit = treatment_regression(data)
    return {
        "estimate": fit.estimate,
        "se": fit.se,
        "t_stat": fit.t_stat,
        "partial_r2": fit.partial_r2,
        "rv": robustness_value(fit, q),
        "rv_alpha": robustness_value(fit, q, alpha),
    }


def adjusted_estimates(
    fit: RegressionFit, grid: np.ndarray | None = None
) -> pd.DataFrame:
    """Estimate after removing the worst-case bias, on a grid of equal strengths."""
    grid = np.linspace(0.0, 0.3, 7) if grid is None else grid
    rows = []
    for r2 in grid:
        bias = bias_bound(fit, r2, r2)
        rows.append(
            {
                "r2": float(r2),
                "bias_bound": bias,
                "adjusted_estimate": fit.estimate - np.sign(fit.estimate) * bias,
            }
        )
    return pd.DataFrame(rows)


def _rss(Z: np.ndarray, v: np.ndarray) -> float:
    Z = np.column_stack([np.ones(len(v)), Z]) if Z.size else np.ones((len(v), 1))
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    r = v - Z @ beta
    return float(r @ r)


def _partial_r2(v: np.ndarray, full: np.ndarray, reduced: np.ndarray) -> float:
    rss_reduced = _rss(reduced, v)
    return (rss_reduced - _rss(full, v)) / rss_reduced if rss_reduced > 0 else 0.0


def benchmark_bounds(
    data: CausalData, kd: float = 1.0, ky: float | None = None
) -> pd.DataFrame:
    r"""Bounds on a confounder ``k`` times as strong as each observed covariate.

    For covariate group :math:`j` let :math:`a = R^2_{T\sim X_j \mid X_{-j}}`
    and :math:`b = R^2_{Y\sim X_j \mid T, X_{-j}}`. A confounder :math:`k_D`
    times as strong as :math:`X_j` in explaining the treatment and
    :math:`k_Y` times as strong in explaining the outcome satisfies
    (Cinelli and Hazlett, 2020, Section 6)

    .. math::

        R^2_{T\sim Z\mid X} = \frac{k_D\, a}{1 - a}, \qquad
        R^2_{Y\sim Z\mid T,X} \le
        \left(\frac{\sqrt{k_Y} + \sqrt{\eta}}{\sqrt{1-\eta}}\right)^2
        \frac{b}{1 - b},
        \qquad \eta = \frac{k_D\, a^2}{(1 - k_D a)(1 - a)}.
    """
    ky = kd if ky is None else ky
    fit = treatment_regression(data)
    blocks = {c: indicator_frame(data.X[[c]]).to_numpy() for c in data.X.columns}
    t = data.t.astype(float)
    rows = []
    for name, block in blocks.items():
        others = [b for c, b in blocks.items() if c != name]
        rest = np.column_stack(others) if others else np.empty((len(t), 0))
        r2_tx = _partial_r2(t, np.column_stack([rest, block]), rest)
        with_t = np.column_stack([t, rest])
        r2_yx = _partial_r2(data.y, np.column_stack([with_t, block]), with_t)
        r2_tz = kd * r2_tx / (1 - r2_tx)
        if r2_tz >= 1:
            raise ValueError(f"kd too large for covariate '{name}'")
        eta = kd * r2_tx**2 / ((1 - kd * r2_tx) * (1 - r2_tx))
        r2_yz = min(
            1.0,
            ((np.sqrt(ky) + np.sqrt(eta)) / np.sqrt(1 - eta)) ** 2
            * r2_yx
            / (1 - r2_yx),
        )
        bias = bias_bound(fit, r2_yz, r2_tz)
        rows.append(
            {
                "benchmark": name,
                "kd": kd,
                "ky": ky,
                "r2_tz": r2_tz,
                "r2_yz": r2_yz,
                "bias_bound": bias,
                "adjusted_estimate": fit.estimate - np.sign(fit.estimate) * bias,
            }
        )
    return pd.DataFrame(rows)
