r"""Estimators of the promotion effect.

Every estimator returns an :class:`EffectEstimate` with a standard error and
a 95% interval, so that the estimators can be compared in one table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from loyalty_spend.causal.data import CausalData
from loyalty_spend.causal.propensity import crossfit_propensity

ESTIMANDS = ("ate", "att")


@dataclass(frozen=True)
class EffectEstimate:
    method: str
    estimand: str
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    n: int

    @property
    def p_value(self) -> float:
        if self.se <= 0 or not np.isfinite(self.se):
            return float("nan")
        return float(2 * stats.norm.sf(abs(self.estimate / self.se)))

    def to_dict(self) -> dict[str, float | str | int]:
        return {**asdict(self), "p_value": self.p_value}


def _check_estimand(estimand: str) -> None:
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}")


def hajek_ipw(y: ArrayLike, t: ArrayLike, e: ArrayLike, estimand: str = "ate") -> float:
    """Normalised inverse probability weighting estimate.

    For the ATE the treated are weighted by ``1/e`` and the controls by
    ``1/(1-e)``; for the ATT the treated keep weight one and the controls
    get ``e/(1-e)``. Normalising the weights within each group (Hajek)
    makes the estimator stable when weights do not average to one.
    """
    _check_estimand(estimand)
    y, t, e = np.asarray(y, float), np.asarray(t, int), np.asarray(e, float)
    if estimand == "ate":
        w1, w0 = t / e, (1 - t) / (1 - e)
    else:
        w1, w0 = t.astype(float), (1 - t) * e / (1 - e)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def _bootstrap(
    stat: Callable[[np.ndarray], float],
    n: int,
    n_boot: int,
    seed: int,
    t: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if t[idx].min() == t[idx].max():
            continue
        draws.append(stat(idx))
    return np.asarray(draws)


def ipw(
    data: CausalData,
    estimand: str = "ate",
    propensity: ArrayLike | None = None,
    n_boot: int = 500,
    seed: int = 0,
) -> EffectEstimate:
    """Hajek IPW with a percentile bootstrap interval.

    The propensity is cross-fitted once and held fixed in the bootstrap.
    Ignoring its estimation error makes the interval conservative for the
    ATE, since an estimated propensity removes chance imbalance that a known
    one would leave (Hirano, Imbens and Ridder, 2003).
    """
    _check_estimand(estimand)
    e = (
        crossfit_propensity(data.X, data.t, seed=seed)
        if propensity is None
        else np.asarray(propensity, float)
    )
    est = hajek_ipw(data.y, data.t, e, estimand)
    draws = _bootstrap(
        lambda i: hajek_ipw(data.y[i], data.t[i], e[i], estimand),
        len(data.y),
        n_boot,
        seed,
        data.t,
    )
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return EffectEstimate(
        "IPW",
        estimand,
        est,
        float(draws.std(ddof=1)),
        float(lo),
        float(hi),
        len(data.y),
    )
