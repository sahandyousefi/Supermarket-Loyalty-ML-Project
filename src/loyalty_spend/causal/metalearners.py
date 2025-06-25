r"""Meta-learners for the conditional average treatment effect.

The CATE :math:`\tau(x) = \mathbb E[Y(1) - Y(0) \mid X = x]` says for whom
the promotion works. Meta-learners build it from ordinary regression
models (Kunzel et al., 2019); they differ in how they use the data and in
how they behave when the treatment groups have different sizes or when the
effect is simpler than the outcome surfaces.

Accuracy is measured by the precision in estimating heterogeneous effects,
:math:`\mathrm{PEHE} = \sqrt{n^{-1}\sum_i (\hat\tau(x_i) - \tau(x_i))^2}`
(Hill, 2011), which needs the true effects and so only exists in simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from loyalty_spend.causal.propensity import design

TREATMENT_COLUMN = "_treatment"


def pehe(tau_hat: ArrayLike, tau: ArrayLike) -> float:
    """Root mean squared error of estimated individual effects."""
    d = np.asarray(tau_hat, float) - np.asarray(tau, float)
    return float(np.sqrt(np.mean(d**2)))


def base_learner(X: pd.DataFrame, seed: int = 0) -> Pipeline:
    """Gradient boosting on the covariate design, the default for every learner."""
    return Pipeline(
        [
            ("design", design(X)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    random_state=seed,
                ),
            ),
        ]
    )


class SLearner:
    """One model with the treatment as an extra feature.

    Regularisation can shrink the treatment split away when the effect is
    small relative to the outcome, which biases the CATE towards zero.
    """

    def __init__(self, learner: BaseEstimator | None = None, seed: int = 0) -> None:
        self.learner = learner
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: ArrayLike, y: ArrayLike) -> SLearner:
        Xt = X.assign(**{TREATMENT_COLUMN: np.asarray(t, float)})
        learner = self.learner or base_learner(Xt, self.seed)
        self.model_ = clone(learner).fit(Xt, np.asarray(y, float))
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        treated = X.assign(**{TREATMENT_COLUMN: 1.0})
        control = X.assign(**{TREATMENT_COLUMN: 0.0})
        return self.model_.predict(treated) - self.model_.predict(control)
