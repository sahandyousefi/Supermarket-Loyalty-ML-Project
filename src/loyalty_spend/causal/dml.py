r"""Double machine learning for the partially linear model.

The model is

.. math::

    Y = \theta T + g(X) + \varepsilon, \qquad T = m(X) + V,

with :math:`\mathbb E[\varepsilon \mid X, T] = 0` and
:math:`\mathbb E[V \mid X] = 0`. Partialling out :math:`\ell(X) = \mathbb
E[Y \mid X]` and :math:`m(X)` with cross-fitted learners and regressing the
residuals on each other gives the Neyman-orthogonal estimate

.. math::

    \hat\theta = \frac{\sum_i \tilde v_i \tilde y_i}{\sum_i \tilde v_i^2},
    \qquad \tilde y_i = y_i - \hat\ell(x_i),\; \tilde v_i = t_i - \hat m(x_i),

whose first-order insensitivity to errors in :math:`\hat\ell` and
:math:`\hat m` gives root-n inference even with flexible learners
(Chernozhukov et al., 2018). With a heterogeneous effect, :math:`\theta`
is a variance-weighted average of the individual effects, with weights
:math:`\mathrm{Var}(T \mid X)`. Repeating the sample split and taking the
median removes the dependence on one random split.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from loyalty_spend.causal.data import CausalData
from loyalty_spend.causal.estimators import EffectEstimate
from loyalty_spend.causal.propensity import design, propensity_model


def default_outcome_learner(data: CausalData, seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("design", design(data.X)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    random_state=seed,
                ),
            ),
        ]
    )


def _one_split(
    data: CausalData,
    outcome: BaseEstimator,
    treatment: BaseEstimator,
    n_folds: int,
    seed: int,
) -> tuple[float, float]:
    X, t, y = data.X, data.t, data.y
    y_res = np.empty(len(y))
    t_res = np.empty(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        y_res[te] = y[te] - clone(outcome).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        m = clone(treatment).fit(X.iloc[tr], t[tr]).predict_proba(X.iloc[te])[:, 1]
        t_res[te] = t[te] - m
    theta = float(np.sum(t_res * y_res) / np.sum(t_res**2))
    psi = (y_res - theta * t_res) * t_res
    se = float(np.sqrt(np.mean(psi**2) / np.mean(t_res**2) ** 2 / len(y)))
    return theta, se


def dml_plr(
    data: CausalData,
    outcome: BaseEstimator | None = None,
    treatment: BaseEstimator | None = None,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 0,
) -> EffectEstimate:
    """Partially linear DML estimate with median aggregation over splits.

    The aggregated variance adds the spread of the split estimates around
    their median, as recommended by Chernozhukov et al. (2018, Section 3.4).
    """
    outcome = outcome if outcome is not None else default_outcome_learner(data, seed)
    treatment = treatment if treatment is not None else propensity_model(data.X)
    fits = [
        _one_split(data, outcome, treatment, n_folds, seed + r)
        for r in range(n_repeats)
    ]
    thetas = np.array([f[0] for f in fits])
    ses = np.array([f[1] for f in fits])
    theta = float(np.median(thetas))
    se = float(np.sqrt(np.median(ses**2 + (thetas - theta) ** 2)))
    z = stats.norm.ppf(0.975)
    return EffectEstimate(
        "DML (partially linear)",
        "ate",
        theta,
        se,
        theta - z * se,
        theta + z * se,
        len(data.y),
    )
