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


def test_posterior_predictive_covers_new_members(train_df):
    from loyalty_spend.features.preprocess import split_xy

    X, y = split_xy(train_df)
    post = fit_hierarchical(
        X.iloc[:700], y.iloc[:700], n_chains=2, n_iter=600, n_warmup=200
    )
    draws = post.predictive(X.iloc[700:], n_draws=800)
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=1)
    covered = (y.iloc[700:].to_numpy() >= lo) & (y.iloc[700:].to_numpy() <= hi)
    assert covered.mean() == pytest.approx(0.9, abs=0.06)
    assert np.corrcoef(post.predict(X.iloc[700:]), y.iloc[700:])[0, 1] > 0.99
    new_group = X.iloc[700:710].assign(region="Antarctica")
    assert np.isfinite(post.predictive(new_group, 50)).all()


def test_group_specific_variances():
    rng = np.random.default_rng(5)
    n = 1500
    band = rng.integers(0, 3, size=n)
    x = rng.normal(size=n)
    sd = np.array([0.5, 1.0, 3.0])[band]
    y = 1 + x + rng.normal(size=n) * sd
    X = pd.DataFrame(
        {
            "x": x,
            "grp": rng.integers(0, 4, size=n).astype(str),
            "band": band.astype(str),
        }
    )
    post = fit_hierarchical(
        X,
        y,
        group="grp",
        fixed=["x"],
        variance_by="band",
        n_chains=2,
        n_iter=500,
        n_warmup=200,
    )
    table = post.summary().set_index("parameter")
    for level, true in zip("012", [0.5, 1.0, 3.0]):
        assert table.loc[f"sigma[{level}]", "mean"] == pytest.approx(true, rel=0.1)
    draws = post.predictive(X.head(300), 400)
    spread = draws.std(axis=1)
    assert (
        spread[X.head(300)["band"].to_numpy() == "2"].mean()
        > 2 * spread[X.head(300)["band"].to_numpy() == "0"].mean()
    )


def test_posterior_sampler_widens_the_cost_distribution(train_df):
    from loyalty_spend.bayes.hierarchical import posterior_sampler
    from loyalty_spend.business.rewards import (
        RewardSchedule,
        gaussian_sampler,
        simulate_liability,
    )
    from loyalty_spend.config import RewardConfig
    from loyalty_spend.features.preprocess import split_xy

    X, y = split_xy(train_df)
    post = fit_hierarchical(X, y, n_chains=2, n_iter=500, n_warmup=200)
    schedule = RewardSchedule(RewardConfig().tiers)
    customers = X.iloc[:200]
    mean = post.predict(customers)
    sigma = float(np.sqrt(post.flat("sigma2")).mean())
    full = simulate_liability(
        range(200), mean, posterior_sampler(post, customers), schedule, 2000, 0
    )
    plug = simulate_liability(
        range(200), mean, gaussian_sampler(mean, sigma), schedule, 2000, 0
    )
    assert full.total_draws.std() > plug.total_draws.std()
    assert full.total_draws.mean() == pytest.approx(plug.total_draws.mean(), rel=0.02)


def test_group_contrasts_sum_to_zero(synthetic):
    X, y, u = synthetic
    post = fit_hierarchical(
        X, y, group="grp", fixed=["x"], n_chains=1, n_iter=400, n_warmup=100
    )
    table = post.group_contrasts()
    assert table["mean"].sum() == pytest.approx(0.0, abs=1e-9)
    assert np.corrcoef(table["mean"], u - u.mean())[0, 1] > 0.98


def test_pooled_variances_help_small_groups_with_a_misscaled_prior():
    # Twelve groups of 15 members with residual SD near 0.2: the default
    # prior scale b0 = 1 pulls every group variance up, a learned scale does not.
    rng = np.random.default_rng(5)
    n = 180
    band = np.repeat(np.arange(12), 15)
    sd = np.linspace(0.15, 0.25, 12)[band]
    x = rng.normal(size=n)
    y = 1 + x + rng.normal(size=n) * sd
    X = pd.DataFrame(
        {"x": x, "grp": rng.integers(0, 3, n).astype(str), "band": band.astype(str)}
    )
    kwargs = dict(group="grp", fixed=["x"], variance_by="band", n_chains=2)
    fixed = fit_hierarchical(X, y, n_iter=600, n_warmup=200, **kwargs)
    pooled = fit_hierarchical(
        X, y, n_iter=600, n_warmup=200, pool_variances=True, **kwargs
    )
    est_fixed = np.sqrt(fixed.flat("sigma2")).mean(axis=0)
    est_pooled = np.sqrt(pooled.flat("sigma2")).mean(axis=0)
    truth = np.linspace(0.15, 0.25, 12)
    assert np.mean(np.abs(est_pooled - truth)) < 0.5 * np.mean(
        np.abs(est_fixed - truth)
    )
    assert pooled.variance_scale is not None and fixed.variance_scale is None
