import numpy as np
import pandas as pd
import pytest

from loyalty_spend.causal.data import CausalData, from_frame
from loyalty_spend.causal.sensitivity import (
    RegressionFit,
    adjusted_estimates,
    bias_bound,
    robustness_value,
    sensitivity_summary,
    treatment_regression,
)
from loyalty_spend.causal.simulate import simulate_promotion


def test_robustness_value_reduces_the_estimate_to_zero():
    # A confounder with both partial R-squared equal to RV_1 produces a bias
    # exactly as large as the estimate.
    fit = RegressionFit(estimate=2.0, se=0.5, df=400)
    rv = robustness_value(fit, q=1.0)
    assert bias_bound(fit, rv, rv) == pytest.approx(2.0, rel=1e-9)
    assert robustness_value(fit, q=1.0, alpha=0.05) < rv
    weak = RegressionFit(estimate=0.1, se=0.5, df=400)
    assert robustness_value(weak, alpha=0.05) == 0.0


def test_partial_r2_matches_its_definition():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.integers(0, 3, size=n)
    t = (rng.uniform(size=n) < 0.3 + 0.1 * x).astype(int)
    y = 2.0 * t + x + rng.normal(size=n)
    data = CausalData(pd.DataFrame({"x": x.astype(str)}), t, y)
    fit = treatment_regression(data)
    assert fit.estimate == pytest.approx(2.0, abs=0.15)
    # Frisch-Waugh: partial R2 from residualising both y and t on x.
    dummies = pd.get_dummies(x).to_numpy(float)

    def resid(v):
        beta, *_ = np.linalg.lstsq(dummies, v, rcond=None)
        return v - dummies @ beta

    ry, rt = resid(y), resid(t.astype(float))
    direct = np.corrcoef(ry, rt)[0, 1] ** 2
    assert fit.partial_r2 == pytest.approx(direct, rel=1e-6)


def test_summary_and_adjusted_grid(train_df):
    data = from_frame(simulate_promotion(train_df, n=2000, effect=6.0, seed=10).frame)
    summary = sensitivity_summary(data)
    assert 0 < summary["rv_alpha"] < summary["rv"] < 1
    grid = adjusted_estimates(treatment_regression(data))
    assert grid["bias_bound"].is_monotonic_increasing
    assert grid.loc[0, "adjusted_estimate"] == pytest.approx(summary["estimate"])
    with pytest.raises(ValueError):
        bias_bound(treatment_regression(data), 0.5, 1.0)
