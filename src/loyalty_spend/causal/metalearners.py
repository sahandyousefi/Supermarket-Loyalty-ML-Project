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


class TLearner:
    """Separate outcome models for treated and control members.

    The CATE is the difference of the two fits, so it inherits the noise of
    both; with a small treated group the treated model is the weak link.
    """

    def __init__(self, learner: BaseEstimator | None = None, seed: int = 0) -> None:
        self.learner = learner
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: ArrayLike, y: ArrayLike) -> TLearner:
        t = np.asarray(t, int)
        y = np.asarray(y, float)
        learner = self.learner or base_learner(X, self.seed)
        self.mu0_ = clone(learner).fit(X[t == 0], y[t == 0])
        self.mu1_ = clone(learner).fit(X[t == 1], y[t == 1])
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        return self.mu1_.predict(X) - self.mu0_.predict(X)


class XLearner:
    """Kunzel et al. (2019): impute individual effects, then smooth them.

    Treated members get the imputed effect ``y - mu0(x)`` and controls get
    ``mu1(x) - y``. A second-stage model is fitted to each set, and the two
    are combined with the propensity as weight,
    ``tau(x) = e(x) tau0(x) + (1 - e(x)) tau1(x)``, which leans on the model
    fitted to the larger group where the other group is sparse.
    """

    def __init__(self, learner: BaseEstimator | None = None, seed: int = 0) -> None:
        self.learner = learner
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: ArrayLike, y: ArrayLike) -> XLearner:
        from loyalty_spend.causal.propensity import propensity_model

        t = np.asarray(t, int)
        y = np.asarray(y, float)
        learner = self.learner or base_learner(X, self.seed)
        X0, X1 = X[t == 0], X[t == 1]
        mu0 = clone(learner).fit(X0, y[t == 0])
        mu1 = clone(learner).fit(X1, y[t == 1])
        d1 = y[t == 1] - mu0.predict(X1)
        d0 = mu1.predict(X0) - y[t == 0]
        self.tau1_ = clone(learner).fit(X1, d1)
        self.tau0_ = clone(learner).fit(X0, d0)
        self.propensity_ = propensity_model(X).fit(X, t)
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        e = self.propensity_.predict_proba(X)[:, 1]
        return e * self.tau0_.predict(X) + (1 - e) * self.tau1_.predict(X)


class DRLearner:
    """Kennedy (2023): regress the doubly robust pseudo-outcome on X.

    The AIPW score of every unit, built from cross-fitted nuisances, has
    conditional mean ``tau(x)`` when either nuisance is correct, so a
    regression of the scores on the covariates estimates the CATE with an
    error that depends on the product of the nuisance errors.
    """

    def __init__(
        self,
        learner: BaseEstimator | None = None,
        n_folds: int = 5,
        seed: int = 0,
    ) -> None:
        self.learner = learner
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: ArrayLike, y: ArrayLike) -> DRLearner:
        from loyalty_spend.causal.data import CausalData
        from loyalty_spend.causal.doubly_robust import aipw_scores, crossfit_nuisance

        data = CausalData(
            X.reset_index(drop=True), np.asarray(t, int), np.asarray(y, float)
        )
        nuisance = crossfit_nuisance(data, n_folds=self.n_folds, seed=self.seed)
        self.pseudo_outcome_ = aipw_scores(data, nuisance)
        learner = self.learner or base_learner(X, self.seed)
        self.model_ = clone(learner).fit(data.X, self.pseudo_outcome_)
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)


class RLearner:
    r"""Nie and Wager (2021): minimise the Robinson residual loss.

    With cross-fitted :math:`\hat\ell(x) = \mathbb E[Y\mid x]` and
    :math:`\hat m(x) = \Pr(T = 1\mid x)`, the CATE minimises
    :math:`\sum_i \big(\tilde y_i - \tau(x_i)\,\tilde t_i\big)^2`, which is a
    regression of :math:`\tilde y_i / \tilde t_i` on :math:`x_i` with weights
    :math:`\tilde t_i^2`. The final model needs ``sample_weight`` support.
    """

    def __init__(
        self,
        learner: BaseEstimator | None = None,
        n_folds: int = 5,
        seed: int = 0,
    ) -> None:
        self.learner = learner
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: ArrayLike, y: ArrayLike) -> RLearner:
        from sklearn.model_selection import KFold

        from loyalty_spend.causal.propensity import propensity_model

        t = np.asarray(t, int)
        y = np.asarray(y, float)
        y_res = np.empty(len(y))
        t_res = np.empty(len(y))
        outcome = base_learner(X, self.seed)
        treatment = propensity_model(X)
        for tr, te in KFold(self.n_folds, shuffle=True, random_state=self.seed).split(
            X
        ):
            y_res[te] = y[te] - clone(outcome).fit(X.iloc[tr], y[tr]).predict(
                X.iloc[te]
            )
            m = clone(treatment).fit(X.iloc[tr], t[tr]).predict_proba(X.iloc[te])[:, 1]
            t_res[te] = t[te] - m
        # Units with almost no treatment residual carry no information about
        # the effect but would get huge pseudo-outcomes.
        t_res = np.where(np.abs(t_res) < 1e-3, np.sign(t_res) * 1e-3 + 1e-12, t_res)
        learner = self.learner or base_learner(X, self.seed)
        self.model_ = clone(learner).fit(
            X, y_res / t_res, model__sample_weight=t_res**2
        )
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)


LEARNERS = {
    "s": SLearner,
    "t": TLearner,
    "x": XLearner,
    "dr": DRLearner,
    "r": RLearner,
}
