"""All promotion-effect estimators on one study, with their diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from loyalty_spend.causal.balance import balance_table
from loyalty_spend.causal.data import CausalData
from loyalty_spend.causal.dml import dml_plr
from loyalty_spend.causal.doubly_robust import aipw, aipw_att, crossfit_nuisance
from loyalty_spend.causal.estimators import EffectEstimate, ipw, regression_adjustment
from loyalty_spend.causal.overlap import overlap_summary
from loyalty_spend.causal.sensitivity import sensitivity_summary


def naive_difference(data: CausalData) -> EffectEstimate:
    """Difference in means with a Welch standard error."""
    y1, y0 = data.y[data.t == 1], data.y[data.t == 0]
    est = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    z = stats.norm.ppf(0.975)
    return EffectEstimate(
        "Difference in means", "ate", est, se, est - z * se, est + z * se, len(data.y)
    )


@dataclass
class PromotionAnalysis:
    estimates: pd.DataFrame
    overlap: dict[str, float]
    balance: pd.DataFrame
    sensitivity: dict[str, float] = field(default_factory=dict)


def analyse_promotion(
    data: CausalData, n_boot: int = 300, n_repeats: int = 3, seed: int = 0
) -> PromotionAnalysis:
    """Estimate the ATE and ATT with every estimator and collect diagnostics."""
    nuisance = crossfit_nuisance(data, seed=seed)
    estimates = [
        naive_difference(data),
        regression_adjustment(data, "ate", n_boot=n_boot, seed=seed),
        ipw(data, "ate", propensity=nuisance.e, n_boot=n_boot, seed=seed),
        aipw(data, nuisance),
        dml_plr(data, n_repeats=n_repeats, seed=seed),
        regression_adjustment(data, "att", n_boot=n_boot, seed=seed),
        ipw(data, "att", propensity=nuisance.e, n_boot=n_boot, seed=seed),
        aipw_att(data, nuisance),
    ]
    table = pd.DataFrame([e.to_dict() for e in estimates])
    return PromotionAnalysis(
        estimates=table,
        overlap=overlap_summary(nuisance.e, data.t),
        balance=balance_table(data.X, data.t, nuisance.e),
        sensitivity=sensitivity_summary(data),
    )
