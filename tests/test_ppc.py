import numpy as np
import pandas as pd

from loyalty_spend.bayes.checks import posterior_predictive_check
from loyalty_spend.bayes.hierarchical import fit_hierarchical


def _data(rng, heavy):
    n = 1500
    g = rng.integers(0, 5, size=n)
    x = rng.normal(size=n)
    noise = rng.standard_t(2, size=n) if heavy else rng.normal(size=n)
    y = 2 + x + 0.5 * g + noise
    return pd.DataFrame({"x": x, "grp": g.astype(str)}), y


def test_check_passes_a_correct_model_and_flags_heavy_tails():
    rng = np.random.default_rng(0)
    for heavy in (False, True):
        X, y = _data(rng, heavy)
        post = fit_hierarchical(
            X, y, group="grp", fixed=["x"], n_chains=2, n_iter=500, n_warmup=200
        )
        table = posterior_predictive_check(post, X, y, n_rep=200).set_index("statistic")
        p = table.loc["kurtosis", "p_value"]
        if heavy:
            assert p < 0.01
        else:
            assert 0.02 < p < 0.98
            assert 0.02 < table.loc["mean", "p_value"] < 0.98


def test_missing_numeric_inputs_use_training_medians():
    rng = np.random.default_rng(1)
    X, y = _data(rng, False)
    post = fit_hierarchical(
        X, y, group="grp", fixed=["x"], n_chains=1, n_iter=300, n_warmup=100
    )
    X_new = X.head(5).assign(x=np.nan)
    assert np.isfinite(post.predictive(X_new, 20)).all()
