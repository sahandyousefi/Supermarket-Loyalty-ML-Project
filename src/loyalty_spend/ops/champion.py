"""Champion-challenger comparison on a labelled holdout.

Both models score the same customers, so their squared errors are paired.
The test is on the per-customer loss differential
:math:`d_i = (y_i - \\hat y^{C}_i)^2 - (y_i - \\hat y^{N}_i)^2`, as in the
Diebold and Mariano (1995) comparison of forecasts; a bootstrap interval for
the RMSE difference gives the size of the gain in the units of spend. A
challenger is promoted only if it is better by more than a stated margin,
so a new model does not replace a working one on noise.

Point accuracy is not the only thing a forecast is used for: the reward
cost forecast and the scoring service both rely on the intervals. A
challenger that is as accurate as the champion but gives better intervals
deserves promotion too. The interval comparison bootstraps the paired
difference in mean interval score (Gneiting and Raftery, 2007), and
``promotion_decision`` promotes a challenger whose RMSE is not worse than
the champion's by more than a non-inferiority margin and whose interval
score is better by more than its own margin.
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


@dataclass(frozen=True)
class IntervalChallengeResult:
    champion_score: float
    challenger_score: float
    score_difference: float
    ci_lower: float
    ci_upper: float
    champion_coverage: float
    challenger_coverage: float


def compare_intervals(
    y: ArrayLike,
    champion: tuple[ArrayLike, ArrayLike],
    challenger: tuple[ArrayLike, ArrayLike],
    alpha: float = 0.1,
    n_boot: int = 2000,
    seed: int = 0,
) -> IntervalChallengeResult:
    """Paired bootstrap of the difference in mean interval score."""
    from loyalty_spend.evaluation.scoring import interval_score

    y = np.asarray(y, float)
    s_old = interval_score(y, *champion, alpha)
    s_new = interval_score(y, *challenger, alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = s_new[idx].mean(axis=1) - s_old[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])

    def coverage(bounds: tuple[ArrayLike, ArrayLike]) -> float:
        lower, upper = (np.asarray(b, float) for b in bounds)
        return float(np.mean((y >= lower) & (y <= upper)))

    return IntervalChallengeResult(
        float(s_old.mean()),
        float(s_new.mean()),
        float(s_new.mean() - s_old.mean()),
        float(lo),
        float(hi),
        coverage(champion),
        coverage(challenger),
    )


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def promotion_decision(
    point: ChallengeResult,
    intervals: IntervalChallengeResult,
    rmse_margin: float = 0.01,
    score_margin: float = 0.0,
) -> PromotionDecision:
    """Promote on clearly better accuracy, or on equal accuracy and better intervals."""
    if point.challenger_wins:
        return PromotionDecision(True, "lower RMSE beyond the margin")
    not_worse = point.ci_upper < rmse_margin
    better_intervals = intervals.ci_upper < -score_margin
    if not_worse and better_intervals:
        return PromotionDecision(
            True, "RMSE not worse by more than the margin and better interval score"
        )
    if not not_worse:
        return PromotionDecision(False, "RMSE may be worse by more than the margin")
    return PromotionDecision(False, "interval score not clearly better")
