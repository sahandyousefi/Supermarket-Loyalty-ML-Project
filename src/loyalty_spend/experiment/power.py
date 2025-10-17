r"""Sample size and power for a difference in mean spend.

For :math:`n_T` treated and :math:`n_C` control members with common
standard deviation :math:`\sigma`, the two-sample :math:`t` statistic has a
noncentral :math:`t` distribution with :math:`\nu = n_T + n_C - 2` degrees
of freedom and noncentrality
:math:`\delta / (\sigma\sqrt{1/n_T + 1/n_C})`. Power is the probability
that it falls beyond the critical value. Covariate adjustment that explains
a share :math:`R^2` of the variance of spend reduces :math:`\sigma^2` to
:math:`\sigma^2(1 - R^2)` and costs one degree of freedom per covariate.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq


def power_two_sample(
    delta: float,
    sd: float,
    n_treat: int,
    n_control: int,
    alpha: float = 0.05,
    two_sided: bool = True,
    r2: float = 0.0,
    n_covariates: int = 0,
) -> float:
    """Power of the (covariate-adjusted) two-sample t-test."""
    if not 0 <= r2 < 1:
        raise ValueError("r2 must be in [0, 1)")
    df = n_treat + n_control - 2 - n_covariates
    if df < 1:
        raise ValueError("too few members for the number of covariates")
    se = sd * np.sqrt((1 - r2) * (1 / n_treat + 1 / n_control))
    nc = delta / se
    if two_sided:
        crit = stats.t.ppf(1 - alpha / 2, df)
        # Far tails of the noncentral t come back as NaN; they are zero.
        upper = np.nan_to_num(stats.nct.sf(crit, df, nc), nan=1.0)
        lower = np.nan_to_num(stats.nct.cdf(-crit, df, nc), nan=0.0)
        return float(min(1.0, upper + lower))
    crit = stats.t.ppf(1 - alpha, df)
    return float(np.nan_to_num(stats.nct.sf(crit, df, nc), nan=1.0))


def sample_size(
    delta: float,
    sd: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
    r2: float = 0.0,
    n_covariates: int = 0,
) -> tuple[int, int]:
    """Smallest control group (and treated group ``ratio`` times as large)."""

    def achieved(n_c: int) -> float:
        n_t = max(2, int(np.ceil(ratio * n_c)))
        return power_two_sample(delta, sd, n_t, n_c, alpha, True, r2, n_covariates)

    lo, hi = 2 + n_covariates, 4 + n_covariates
    while achieved(hi) < power:
        lo, hi = hi, hi * 2
        if hi > 10**9:
            raise ValueError("power cannot be reached")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if achieved(mid) >= power:
            hi = mid
        else:
            lo = mid
    return int(np.ceil(ratio * hi)), hi


def minimum_detectable_effect(
    sd: float,
    n_treat: int,
    n_control: int,
    power: float = 0.8,
    alpha: float = 0.05,
    r2: float = 0.0,
    n_covariates: int = 0,
) -> float:
    """Smallest difference detected with the given power."""
    se = sd * np.sqrt(1 / n_treat + 1 / n_control)
    return float(
        brentq(
            lambda d: (
                power_two_sample(
                    d, sd, n_treat, n_control, alpha, True, r2, n_covariates
                )
                - power
            ),
            1e-9 * se,
            15 * se,
        )
    )
