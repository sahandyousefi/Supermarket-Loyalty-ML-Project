import numpy as np
import pandas as pd
import pytest

from loyalty_spend.analysis.drivers import freedman_lane


def _data(rng, n=200, effect=0.0, heavy=False):
    a = rng.choice(["x", "y", "z"], n)
    b = rng.choice(["p", "q"], n)
    noise = rng.standard_t(3, n) if heavy else rng.normal(size=n)
    y = np.where(b == "q", 2.0, 0.0) + effect * (a == "z") + noise
    return pd.DataFrame({"a": a, "b": b, "spend": y})


def test_matches_the_f_test_with_normal_errors():
    df = _data(np.random.default_rng(0), effect=0.4)
    res = freedman_lane(df, ["a"], ["b"], n_perm=2000)
    assert res["df1"] == 2 and res["df2"] == 200 - 4
    assert res["p_permutation"] == pytest.approx(res["p_f"], abs=0.02)


def test_level_under_the_null_with_heavy_tails():
    rng = np.random.default_rng(1)
    p = [
        freedman_lane(_data(rng, 60, heavy=True), ["a"], ["b"], n_perm=199)[
            "p_permutation"
        ]
        for _ in range(200)
    ]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.035)


def test_tenure_effect_given_the_other_attributes(train_df):
    others = ["region", "joining_month", "promotion"]
    res = freedman_lane(train_df, ["loyalty_years"], others, n_perm=199)
    assert res["p_permutation"] == pytest.approx(1 / 200)
    assert res["f"] > 1000
