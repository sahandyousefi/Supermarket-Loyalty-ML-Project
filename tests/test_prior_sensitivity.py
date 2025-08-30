import numpy as np
import pandas as pd

from loyalty_spend.bayes.prior_sensitivity import prior_sensitivity


def test_residual_scale_is_robust_and_group_scale_is_not():
    rng = np.random.default_rng(0)
    n = 1000
    g = rng.integers(0, 4, size=n)
    x = rng.normal(size=n)
    y = 1 + x + np.array([-0.2, 0.0, 0.1, 0.3])[g] + rng.normal(size=n)
    X = pd.DataFrame({"x": x, "grp": g.astype(str)})
    table = prior_sensitivity(
        X, y, group="grp", fixed=["x"], n_chains=1, n_iter=600, n_warmup=200
    )
    assert len(table) == 3
    sigma, tau = table["sigma_mean"], table["tau_mean"]
    assert sigma.max() / sigma.min() < 1.05
    assert tau.max() / tau.min() > 1.5
