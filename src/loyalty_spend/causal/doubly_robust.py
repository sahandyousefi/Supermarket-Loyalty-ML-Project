r"""Doubly robust estimation with cross-fitting.

The augmented inverse probability weighting (AIPW) score of the ATE is

.. math::

    \psi_i = \hat\mu_1(x_i) - \hat\mu_0(x_i)
    + \frac{t_i\,(y_i - \hat\mu_1(x_i))}{\hat e(x_i)}
    - \frac{(1 - t_i)(y_i - \hat\mu_0(x_i))}{1 - \hat e(x_i)},

and the estimate is the mean of :math:`\psi_i` (Robins, Rotnitzky and Zhao,
1994). It is consistent if either the outcome models or the propensity model
is correct, and when both converge fast enough its error is asymptotically
normal with variance estimated by the empirical variance of the scores.
Nuisance functions are fitted on the other folds (cross-fitting), so that
flexible learners can be used without overfitting bias (Chernozhukov et al.,
2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from loyalty_spend.causal.data import CausalData
from loyalty_spend.causal.estimators import EffectEstimate, outcome_model
from loyalty_spend.causal.propensity import propensity_model


@dataclass
class Nuisance:
    """Out-of-fold propensity and outcome predictions for every unit."""

    e: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray


def crossfit_nuisance(
    data: CausalData,
    outcome: BaseEstimator | None = None,
    propensity: BaseEstimator | None = None,
    n_folds: int = 5,
    seed: int = 0,
    clip: float = 0.01,
) -> Nuisance:
    X, t, y = data.X, data.t, data.y
    outcome = outcome if outcome is not None else outcome_model(X)
    propensity = propensity if propensity is not None else propensity_model(X)
    e = np.empty(len(y))
    mu0 = np.empty(len(y))
    mu1 = np.empty(len(y))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in folds.split(X, t):
        Xtr, ttr, ytr = X.iloc[tr], t[tr], y[tr]
        e[te] = clone(propensity).fit(Xtr, ttr).predict_proba(X.iloc[te])[:, 1]
        mu0[te] = clone(outcome).fit(Xtr[ttr == 0], ytr[ttr == 0]).predict(X.iloc[te])
        mu1[te] = clone(outcome).fit(Xtr[ttr == 1], ytr[ttr == 1]).predict(X.iloc[te])
    return Nuisance(np.clip(e, clip, 1 - clip), mu0, mu1)


def _estimate(
    method: str, estimand: str, scores: np.ndarray, value: float
) -> EffectEstimate:
    n = len(scores)
    se = float(scores.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(0.975)
    return EffectEstimate(
        method, estimand, value, se, value - z * se, value + z * se, n
    )


def aipw_scores(data: CausalData, nuisance: Nuisance) -> np.ndarray:
    t, y = data.t, data.y
    e, mu0, mu1 = nuisance.e, nuisance.mu0, nuisance.mu1
    return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)


def aipw(
    data: CausalData,
    nuisance: Nuisance | None = None,
    seed: int = 0,
) -> EffectEstimate:
    """Cross-fitted AIPW estimate of the ATE with an influence-function error."""
    nuisance = nuisance or crossfit_nuisance(data, seed=seed)
    psi = aipw_scores(data, nuisance)
    return _estimate("AIPW", "ate", psi, float(psi.mean()))


def aipw_att(
    data: CausalData,
    nuisance: Nuisance | None = None,
    seed: int = 0,
) -> EffectEstimate:
    r"""Doubly robust estimate of the effect on the treated.

    With :math:`p = \Pr(T = 1)`, the estimate is

    .. math::

        \hat\tau_{\mathrm{ATT}} = \frac{1}{n p} \sum_i \Big[t_i (y_i - \hat\mu_0(x_i))
        - (1 - t_i)\frac{\hat e(x_i)}{1 - \hat e(x_i)}(y_i - \hat\mu_0(x_i))\Big],

    which needs only the control outcome model. The standard error comes from
    the influence function :math:`(\phi_i - t_i \hat\tau) / p`.
    """
    nuisance = nuisance or crossfit_nuisance(data, seed=seed)
    t, y = data.t, data.y
    e, mu0 = nuisance.e, nuisance.mu0
    p = t.mean()
    phi = t * (y - mu0) - (1 - t) * e / (1 - e) * (y - mu0)
    tau = float(phi.mean() / p)
    influence = (phi - t * tau) / p
    return _estimate("AIPW", "att", influence, tau)


def frame_of(estimates: list[EffectEstimate]) -> pd.DataFrame:
    return pd.DataFrame([e.to_dict() for e in estimates])
