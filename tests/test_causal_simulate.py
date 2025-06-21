import numpy as np
import pandas as pd
import pytest

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.simulate import simulate_promotion


def test_known_effect_and_treated_share(train_df):
    study = simulate_promotion(train_df, n=4000, effect=6.0, confounding=0.0, seed=1)
    assert study.true_ate == pytest.approx(6.0)
    data = from_frame(study.frame)
    assert data.t.mean() == pytest.approx(0.4, abs=0.03)
    # Without confounding the naive difference is unbiased.
    assert data.naive_difference() == pytest.approx(6.0, abs=1.5)


def test_confounding_biases_the_naive_difference(train_df):
    study = simulate_promotion(train_df, n=4000, effect=6.0, confounding=1.5, seed=2)
    data = from_frame(study.frame)
    assert data.naive_difference() > 6.0 + 5.0
    treated = study.frame["promotion"].to_numpy() == "Yes"
    assert study.propensity[treated].mean() > study.propensity[~treated].mean()
    assert np.all((study.propensity > 0) & (study.propensity < 1))


def test_effect_varies_with_tenure(train_df):
    study = simulate_promotion(train_df, n=3000, effect=5.0, heterogeneity=2.0, seed=3)
    bands = study.frame["loyalty_years"].astype(str)
    by_band = pd.Series(study.effect).groupby(bands.to_numpy()).mean()
    assert by_band["0-1"] == pytest.approx(1.0)
    assert by_band["10+"] == pytest.approx(9.0)
    assert by_band["3-5"] == pytest.approx(5.0)
