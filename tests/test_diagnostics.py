import numpy as np
import pytest

from loyalty_spend.bayes.diagnostics import effective_sample_size, split_rhat


def _ar1(rng, phi, chains=4, n=2000, shift=None):
    x = np.zeros((chains, n))
    for c in range(chains):
        e = rng.normal(size=n)
        for t in range(1, n):
            x[c, t] = phi * x[c, t - 1] + e[t]
        if shift is not None:
            x[c] += shift[c]
    return x


def test_independent_chains_have_rhat_near_one_and_full_ess():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 2000))
    assert split_rhat(x) == pytest.approx(1.0, abs=0.01)
    assert effective_sample_size(x) == pytest.approx(8000, rel=0.15)


def test_autocorrelation_reduces_ess_by_the_theoretical_factor():
    rng = np.random.default_rng(1)
    x = _ar1(rng, 0.8)
    # For AR(1), ESS / N = (1 - phi) / (1 + phi) = 1/9.
    assert effective_sample_size(x) / x.size == pytest.approx(1 / 9, rel=0.3)


def test_chains_stuck_in_different_places_fail_rhat():
    rng = np.random.default_rng(2)
    x = _ar1(rng, 0.5, shift=[0.0, 0.0, 3.0, -3.0])
    assert split_rhat(x) > 1.5


def test_posterior_summary_of_the_sampler(train_df):
    from loyalty_spend.bayes.hierarchical import fit_hierarchical
    from loyalty_spend.features.preprocess import split_xy

    X, y = split_xy(train_df)
    post = fit_hierarchical(X, y, n_chains=2, n_iter=600, n_warmup=200)
    table = post.summary().set_index("parameter")
    assert (table["rhat"] < 1.05).all()
    assert table.loc["sigma", "ess"] > 200
