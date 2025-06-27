"""Who responds to the promotion, and whom to offer it to."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from loyalty_spend.causal.data import CausalData
from loyalty_spend.causal.doubly_robust import aipw_scores, crossfit_nuisance
from loyalty_spend.causal.heterogeneity import blp_test, clan, crossfit_cate, gates
from loyalty_spend.causal.metalearners import LEARNERS
from loyalty_spend.causal.policy import budget_curve, compare_policies
from loyalty_spend.causal.uplift import dr_gain_curve, qini_coefficient


@dataclass
class TargetingAnalysis:
    learner: str
    blp: dict[str, float]
    gates: pd.DataFrame
    gates_test: dict[str, float]
    profile: pd.DataFrame
    policies: pd.DataFrame
    budget: pd.DataFrame
    qini: float


def analyse_targeting(
    data: CausalData,
    learner: str = "dr",
    cost: float = 0.0,
    margin: float = 1.0,
    n_folds: int = 5,
    n_groups: int = 5,
    seed: int = 0,
) -> TargetingAnalysis:
    """Cross-fitted CATE, heterogeneity tests and policy values on one study.

    The CATE proxy and the doubly robust scores are both out of fold, so the
    tests and policy values are not evaluated on the data the proxy was
    trained on.
    """
    if learner not in LEARNERS:
        raise KeyError(f"unknown learner '{learner}'; choose from {sorted(LEARNERS)}")
    proxy = crossfit_cate(LEARNERS[learner], data.X, data.t, data.y, n_folds, seed)
    psi = aipw_scores(data, crossfit_nuisance(data, n_folds=n_folds, seed=seed))
    g = gates(proxy, psi, n_groups)
    groups = g.pop("groups")
    assert isinstance(groups, pd.DataFrame)
    gain = dr_gain_curve(proxy, psi, seed=seed)
    return TargetingAnalysis(
        learner=learner,
        blp=blp_test(proxy, psi),
        gates=groups,
        gates_test={k: float(v) for k, v in g.items()},  # type: ignore[arg-type]
        profile=clan(data.X, proxy, n_groups),
        policies=compare_policies(proxy, psi, cost, margin),
        budget=budget_curve(proxy, psi, cost, margin),
        qini=qini_coefficient(gain, len(psi)),
    )
