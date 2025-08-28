import numpy as np
import pandas as pd
import pytest

from loyalty_spend.bayes.hierarchical import fit_hierarchical


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n, G = 1200, 8
    g = rng.integers(0, G, size=n)
    x = rng.normal(size=n)
    u = rng.normal(scale=2.0, size=G)
    y = 5.0 + 3.0 * x + u[g] + rng.normal(scale=1.0, size=n)
    X = pd.DataFrame({"x": x, "grp": [f"g{i}" for i in g]})
    return X, y, u


def test_gibbs_recovers_the_parameters(synthetic):
    X, y, u = synthetic
    post = fit_hierarchical(
        X, y, group="grp", fixed=["x"], n_chains=2, n_iter=800, n_warmup=300
    )
    beta = post.flat("beta")
    slope = beta[:, post.feature_names.index("num__x")]
    assert slope.mean() == pytest.approx(3.0 * np.std(X["x"]), abs=0.1)
    assert np.sqrt(post.flat("sigma2")).mean() == pytest.approx(1.0, abs=0.08)
    effects = post.flat("u").mean(axis=0) + beta[:, 0].mean() - 5.0
    assert np.corrcoef(effects, u)[0, 1] > 0.98
    assert post.u.shape == (2, 500, 8)
