import numpy as np
import pandas as pd
import pytest

from loyalty_spend.bayes.compare import cv_log_scores, waic
from loyalty_spend.bayes.hierarchical import fit_hierarchical
from loyalty_spend.evaluation.cv import make_splits


def test_waic_of_a_known_model():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    # A point-mass posterior at the truth: no effective parameters.
    ll = np.tile(-0.5 * np.log(2 * np.pi) - 0.5 * y**2, (50, 1))
    out = waic(ll)
    assert out["p_waic"] == pytest.approx(0.0, abs=1e-9)
    assert out["elpd_waic"] == pytest.approx(ll[0].sum())


def test_heavy_tails_favour_the_student_model_out_of_sample():
    rng = np.random.default_rng(1)
    n = 900
    g = rng.integers(0, 4, size=n)
    x = rng.normal(size=n)
    y = 1 + x + 0.3 * g + rng.standard_t(2, size=n)
    X = pd.DataFrame({"x": x, "grp": g.astype(str)})
    fixed = ["x"]
    models = {
        "normal": lambda Xt, yt: fit_hierarchical(
            Xt, yt, "grp", fixed, n_chains=1, n_iter=400, n_warmup=150
        ),
        "student": lambda Xt, yt: fit_hierarchical(
            Xt, yt, "grp", fixed, nu=2.0, n_chains=1, n_iter=400, n_warmup=150
        ),
    }
    table = cv_log_scores(
        X, y, models, make_splits(n, 3, 1, 0), ols_columns=["x", "grp"]
    )
    means = table.groupby("model")["log_score"].mean()
    assert means["student"] > means["normal"]
    assert means["normal"] == pytest.approx(means["ols"], abs=0.05)
    post = models["student"](X, y)
    assert np.isfinite(waic(post.log_likelihood(X, y, max_draws=200))["waic"])
