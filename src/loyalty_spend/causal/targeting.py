"""Who responds to the promotion, and whom to offer it to."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from loyalty_spend.causal.data import CausalData
from loyalty_spend.causal.doubly_robust import aipw_scores, split_nuisance
from loyalty_spend.causal.heterogeneity import blp_test, clan, gates
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


def _halves(
    data: CausalData, rng: np.random.Generator
) -> tuple[CausalData, CausalData]:
    """Random halves with the treated share preserved in each."""
    main = np.zeros(len(data.t), bool)
    for arm in (0, 1):
        idx = np.flatnonzero(data.t == arm)
        main[rng.permutation(idx)[: len(idx) // 2]] = True

    def subset(mask: np.ndarray) -> CausalData:
        return CausalData(
            data.X[mask].reset_index(drop=True), data.t[mask], data.y[mask]
        )

    return subset(~main), subset(main)


def _median_frame(frames: list[pd.DataFrame], key: str) -> pd.DataFrame:
    return pd.concat(frames).groupby(key, sort=False).median().reset_index()


def analyse_targeting(
    data: CausalData,
    learner: str = "dr",
    cost: float = 0.0,
    margin: float = 1.0,
    n_splits: int = 10,
    n_groups: int = 5,
    seed: int = 0,
) -> TargetingAnalysis:
    """Heterogeneity tests and policy values by repeated sample splitting.

    In each split the CATE learner and the nuisance models are trained on
    one half and everything is evaluated on the other, so the proxy and the
    doubly robust scores of the evaluation half are independent. Nested
    cross-fitting on the full sample does not have this property: a unit's
    score and its neighbours' proxies share training data, which biases the
    heterogeneity test. Estimates are the medians over splits and p-values
    are twice the median p-value, capped at one (Chernozhukov et al., 2018).
    """
    if learner not in LEARNERS:
        raise KeyError(f"unknown learner '{learner}'; choose from {sorted(LEARNERS)}")
    rng = np.random.default_rng(seed)
    blps, tests, groups, profiles, policies, budgets, qinis = [], [], [], [], [], [], []
    for _ in range(n_splits):
        aux, main = _halves(data, rng)
        model = LEARNERS[learner]().fit(aux.X, aux.t, aux.y)
        proxy = model.effect(main.X)
        psi = aipw_scores(main, split_nuisance(aux, main))
        blps.append(blp_test(proxy, psi))
        g = gates(proxy, psi, n_groups)
        groups.append(g.pop("groups"))
        tests.append(g)
        profiles.append(clan(main.X, proxy, n_groups))
        policies.append(compare_policies(proxy, psi, cost, margin))
        budgets.append(budget_curve(proxy, psi, cost, margin))
        qinis.append(qini_coefficient(dr_gain_curve(proxy, psi, seed=seed), len(psi)))

    def median_dict(items: list[dict]) -> dict[str, float]:
        frame = pd.DataFrame(items)
        out = {k: float(v) for k, v in frame.median().items()}
        for k in [c for c in frame.columns if c.endswith("p_value")]:
            out[k] = float(min(1.0, 2 * frame[k].median()))
        return out

    return TargetingAnalysis(
        learner=learner,
        blp=median_dict(blps),
        gates=_median_frame(groups, "group"),
        gates_test=median_dict(tests),
        profile=_median_frame(profiles, "covariate"),
        policies=_median_frame(policies, "policy"),
        budget=_median_frame(budgets, "budget_share"),
        qini=float(np.median(qinis)),
    )
