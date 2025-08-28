import numpy as np
import pytest

from loyalty_spend.bayes.empirical import shrink_group_means


def _groups(rng, n_groups=40, size=4, tau=1.0, sigma=3.0):
    theta = rng.normal(10, tau, size=n_groups)
    g = np.repeat(np.arange(n_groups), size)
    y = theta[g] + rng.normal(scale=sigma, size=g.size)
    return theta, g, y


def test_shrinkage_lowers_the_error_of_small_groups():
    rng = np.random.default_rng(0)
    raw_err, eb_err = [], []
    for _ in range(20):
        theta, g, y = _groups(rng)
        table = shrink_group_means(y, g)
        raw_err.append(np.mean((table["raw_mean"] - theta) ** 2))
        eb_err.append(np.mean((table["shrunk_mean"] - theta) ** 2))
    assert np.mean(eb_err) < 0.7 * np.mean(raw_err)


def test_weights_grow_with_group_size_and_vanish_without_heterogeneity():
    rng = np.random.default_rng(1)
    g = np.r_[np.zeros(200), np.ones(5), np.full(50, 2)].astype(int)
    y = np.r_[rng.normal(0, 1, 200), rng.normal(3, 1, 5), rng.normal(-2, 1, 50)]
    table = shrink_group_means(y, g).set_index("group")
    assert table.loc[0, "weight"] > table.loc[2, "weight"] > table.loc[1, "weight"]
    flat = shrink_group_means(rng.normal(size=300), np.repeat(np.arange(30), 10))
    assert flat.attrs["tau2"] == 0.0
    assert flat["shrunk_mean"].nunique() == 1


def test_group_means_of_the_supplied_data(train_df):
    table = shrink_group_means(train_df["spend"], train_df["loyalty_years"])
    assert len(table) == 5
    # Tenure groups are large and far apart, so shrinkage is negligible.
    assert (table["weight"] > 0.99).all()
    assert table["shrunk_mean"].to_numpy() == pytest.approx(
        table["raw_mean"].to_numpy(), abs=0.05
    )
