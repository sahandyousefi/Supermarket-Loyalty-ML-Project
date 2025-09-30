import numpy as np
import pandas as pd
import pytest

from loyalty_spend.synthetic.conditional import ConditionalCopulaSynthesizer
from loyalty_spend.synthetic.copula import StratifiedCopulaSynthesizer
from loyalty_spend.synthetic.fidelity import fidelity_summary, pairwise_fidelity


@pytest.fixture(scope="module")
def data(train_df):
    return train_df.drop(columns="customer_id")


def test_schema_and_ranges(data):
    synth = ConditionalCopulaSynthesizer("loyalty_years", seed=0).fit(data)
    out = synth.sample(2000)
    assert list(out.columns) == list(data.columns)
    for col in ("spend", "first_month", "items_in_first_month"):
        assert out[col].between(data[col].min(), data[col].max()).all()
    assert (out["items_in_first_month"] % 1 == 0).all()
    assert out["joining_month"].dtype == data["joining_month"].dtype


def test_keeps_month_and_tenure_effects(data):
    synth = ConditionalCopulaSynthesizer("loyalty_years", seed=0).fit(data)
    out = synth.sample(len(data))
    pairs = pairwise_fidelity(data, out).set_index("pair")
    assert pairs.loc["items_in_first_month | joining_month", "difference"] < 0.1
    assert pairs.loc["spend | loyalty_years", "difference"] < 0.05
    strat = StratifiedCopulaSynthesizer("loyalty_years", seed=0).fit(data)
    before = fidelity_summary(data, strat.sample(len(data)))
    after = fidelity_summary(data, out)
    assert after["max_pair_difference"] < before["max_pair_difference"] / 3


def test_missing_amounts_are_generated():
    rng = np.random.default_rng(0)
    g = rng.choice(["a", "b"], size=1500)
    x = np.where(g == "a", 10.0, 20.0) + rng.normal(size=1500)
    x[rng.uniform(size=1500) < 0.1] = np.nan
    frame = pd.DataFrame({"g": pd.Categorical(g), "x": x})
    out = ConditionalCopulaSynthesizer(seed=1).fit(frame).sample(4000)
    assert out["x"].isna().mean() == pytest.approx(0.1, abs=0.02)
    means = out.groupby("g", observed=True)["x"].mean()
    assert means["b"] - means["a"] == pytest.approx(10.0, abs=0.3)
