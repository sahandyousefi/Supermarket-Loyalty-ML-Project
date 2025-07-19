"""Champion-challenger comparison on a labelled holdout.

Both models score the same customers, so their squared errors are paired.
The test is on the per-customer loss differential
:math:`d_i = (y_i - \\hat y^{C}_i)^2 - (y_i - \\hat y^{N}_i)^2`, as in the
Diebold and Mariano (1995) comparison of forecasts; a bootstrap interval for
the RMSE difference gives the size of the gain in the units of spend. A
challenger is promoted only if it is better by more than a stated margin,
so a new model does not replace a working one on noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from loyalty_spend.models.persistence import ModelBundle


@dataclass(frozen=True)
class ChallengeResult:
    champion_rmse: float
    challenger_rmse: float
    rmse_difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    margin: float

    @property
    def challenger_wins(self) -> bool:
        """The whole interval of the RMSE difference lies below ``-margin``."""
        return self.ci_upper < -self.margin


def compare_predictions(
    y: ArrayLike,
    champion: ArrayLike,
    challenger: ArrayLike,
    margin: float = 0.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> ChallengeResult:
    y = np.asarray(y, float)
    e_old = (y - np.asarray(champion, float)) ** 2
    e_new = (y - np.asarray(challenger, float)) ** 2
    d = e_old - e_new
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if np.any(d != d[0]) else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = np.sqrt(e_new[idx].mean(axis=1)) - np.sqrt(e_old[idx].mean(axis=1))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    rmse_old, rmse_new = float(np.sqrt(e_old.mean())), float(np.sqrt(e_new.mean()))
    return ChallengeResult(
        rmse_old, rmse_new, rmse_new - rmse_old, float(lo), float(hi), p, margin
    )


def challenge(
    champion: ModelBundle,
    challenger: ModelBundle,
    X: pd.DataFrame,
    y: ArrayLike,
    margin: float = 0.0,
    seed: int = 0,
) -> ChallengeResult:
    """Score both bundles on ``X`` and compare their errors."""
    return compare_predictions(
        y, champion.predict(X), challenger.predict(X), margin=margin, seed=seed
    )
