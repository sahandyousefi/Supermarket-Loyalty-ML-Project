import numpy as np
import pandas as pd
import pytest

from loyalty_spend.synthetic.copula import (
    GaussianCopulaSynthesizer,
    StratifiedCopulaSynthesizer,
)
from loyalty_spend.synthetic.fidelity import (
    correlation_ratio,
    cramers_v,
    fidelity_summary,
    marginal_fidelity,
    pairwise_fidelity,
)


def test_cramers_v_bounds():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice(list("abc"), size=3000))
    assert cramers_v(a, a) == pytest.approx(1.0, abs=1e-3)
    b = pd.Series(rng.choice(list("xyz"), size=3000))
    assert cramers_v(a, b) < 0.05


def test_correlation_ratio_matches_anova_r2():
    rng = np.random.default_rng(1)
    g = pd.Series(rng.choice(["p", "q", "r"], size=500))
    v = pd.Series(g.map({"p": 0.0, "q": 1.0, "r": 2.0}) + rng.normal(size=500))
    frame = pd.DataFrame({"v": v, "g": g})
    means = frame.groupby("g")["v"].transform("mean")
    r2 = 1 - ((v - means) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    assert correlation_ratio(v, g) ** 2 == pytest.approx(r2)


def test_identical_data_has_perfect_fidelity(train_df):
    data = train_df.drop(columns="customer_id")
    marg = marginal_fidelity(data, data)
    assert (marg["value"] == 0).all()
    assert (pairwise_fidelity(data, data)["difference"] < 1e-12).all()


def test_stratified_copula_has_higher_fidelity(train_df):
    data = train_df.drop(columns="customer_id")
    plain = GaussianCopulaSynthesizer(seed=0).fit(data).sample(len(data))
    strat = StratifiedCopulaSynthesizer("loyalty_years", seed=0).fit(data)
    a = fidelity_summary(data, plain)
    b = fidelity_summary(data, strat.sample(len(data)))
    assert b["max_pair_difference"] < a["max_pair_difference"]
    assert b["max_tvd"] < 0.1 and b["max_ks"] < 0.1
