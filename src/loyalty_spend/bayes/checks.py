"""Posterior predictive checks (Gelman et al., 2013, ch. 6).

Replicated data sets are drawn from the posterior predictive distribution
at the observed covariates, and summary statistics of the observed spend
are compared with their distribution over the replications. A posterior
predictive p-value near 0 or 1 marks a feature of the data the model does
not reproduce, for example heavier tails than a normal error allows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from loyalty_spend.bayes.hierarchical import Posterior

Statistic = Callable[[np.ndarray], float]

STATISTICS: dict[str, Statistic] = {
    "mean": lambda y: float(np.mean(y)),
    "sd": lambda y: float(np.std(y, ddof=1)),
    "min": lambda y: float(np.min(y)),
    "max": lambda y: float(np.max(y)),
    "skewness": lambda y: float(stats.skew(y)),
    "kurtosis": lambda y: float(stats.kurtosis(y)),
}


def posterior_predictive_check(
    posterior: Posterior,
    X: pd.DataFrame,
    y: ArrayLike,
    n_rep: int = 500,
    statistics: dict[str, Statistic] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Observed statistic, its replicated distribution and the p-value."""
    statistics = statistics or STATISTICS
    y = np.asarray(y, float)
    reps = posterior.predictive(X, n_draws=n_rep, seed=seed)
    rows = []
    for name, stat in statistics.items():
        observed = stat(y)
        replicated = np.array([stat(reps[:, j]) for j in range(n_rep)])
        rows.append(
            {
                "statistic": name,
                "observed": observed,
                "replicated_mean": float(replicated.mean()),
                "replicated_q05": float(np.quantile(replicated, 0.05)),
                "replicated_q95": float(np.quantile(replicated, 0.95)),
                "p_value": float(np.mean(replicated >= observed)),
            }
        )
    return pd.DataFrame(rows)
