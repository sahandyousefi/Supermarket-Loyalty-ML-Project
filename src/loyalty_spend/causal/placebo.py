"""Placebo test with permuted treatment labels.

If the promotion labels are shuffled, any effect an estimator still finds is
an artefact of the estimator or the data, not of the promotion. The observed
estimate is compared with the distribution of estimates under random
relabelling; the permutation p-value is the share of placebo estimates at
least as large in absolute value. Shuffling within strata of a covariate
keeps the association between that covariate and the treatment, which
tests the estimator against confounding by the other covariates only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from loyalty_spend.arrays import FloatArray
from loyalty_spend.causal.data import CausalData
from loyalty_spend.causal.estimators import _g_computation, outcome_model

Estimator = Callable[[CausalData], float]


def regression_point_estimate(data: CausalData) -> float:
    return _g_computation(data.X, data.t, data.y, "ate", outcome_model(data.X))


@dataclass
class PlaceboResult:
    observed: float
    placebo: FloatArray

    @property
    def p_value(self) -> float:
        extreme = np.sum(np.abs(self.placebo) >= abs(self.observed))
        return float((1 + extreme) / (1 + len(self.placebo)))

    @property
    def placebo_sd(self) -> float:
        return float(self.placebo.std(ddof=1))


def placebo_test(
    data: CausalData,
    estimator: Estimator = regression_point_estimate,
    n_permutations: int = 200,
    strata: str | None = None,
    seed: int = 0,
) -> PlaceboResult:
    rng = np.random.default_rng(seed)
    observed = estimator(data)
    groups = (
        data.X[strata].astype(str).to_numpy()
        if strata is not None
        else np.zeros(len(data.t))
    )
    draws = np.empty(n_permutations)
    for b in range(n_permutations):
        t = data.t.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            t[idx] = rng.permutation(t[idx])
        draws[b] = estimator(CausalData(data.X, t, data.y))
    return PlaceboResult(observed, draws)
