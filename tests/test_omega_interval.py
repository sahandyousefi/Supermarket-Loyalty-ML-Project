import numpy as np
import pandas as pd
import pytest

from loyalty_spend.analysis.drivers import omega_squared_interval


def _frame(rng, means, n=30):
    g = np.repeat(np.arange(len(means)).astype(str), n)
    y = np.concatenate([rng.normal(m, 1.0, n) for m in means])
    return pd.DataFrame({"g": g, "spend": y})


def test_interval_covers_the_population_value():
    means = np.array([0.0, 0.4, 0.8])
    # Population omega squared of equal groups: var(means) / (var(means) + 1).
    true = means.var() / (means.var() + 1.0)
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(300):
        res = omega_squared_interval(_frame(rng, means), "g")
        hits += res["low"] <= true <= res["high"]
    assert hits / 300 == pytest.approx(0.95, abs=0.04)


def test_null_effect_gives_a_lower_bound_of_zero():
    res = omega_squared_interval(_frame(np.random.default_rng(1), [0, 0, 0]), "g")
    assert res["low"] == 0.0 and res["high"] > 0.0


def test_large_effect_on_the_loyalty_data(loyalty_df):
    res = omega_squared_interval(loyalty_df, "loyalty_years")
    assert res["low"] < res["omega_squared"] < res["high"]
    assert res["high"] - res["low"] < 0.02
