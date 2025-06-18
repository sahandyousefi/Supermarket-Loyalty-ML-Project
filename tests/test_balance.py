import numpy as np
import pandas as pd
import pytest

from loyalty_spend.causal.balance import (
    ate_weights,
    att_weights,
    balance_table,
    indicator_frame,
    standardized_mean_differences,
)
from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.propensity import crossfit_propensity
from loyalty_spend.causal.simulate import simulate_promotion


def test_indicator_frame_expands_categories():
    X = pd.DataFrame({"a": ["x", "y", "x"], "b": [1, 2, 3]})
    assert list(indicator_frame(X).columns) == ["a=x", "a=y", "b"]


def test_smd_of_a_known_shift():
    rng = np.random.default_rng(0)
    t = np.r_[np.ones(5000), np.zeros(5000)].astype(int)
    x = np.r_[rng.normal(0.5, 1, 5000), rng.normal(0.0, 1, 5000)]
    smd = standardized_mean_differences(pd.DataFrame({"x": x}), t)
    assert smd["x"] == pytest.approx(0.5, abs=0.05)


def test_weighting_removes_confounding_imbalance(train_df):
    study = simulate_promotion(train_df, n=4000, confounding=1.5, seed=4)
    data = from_frame(study.frame)
    e = crossfit_propensity(data.X, data.t, seed=0)
    table = balance_table(data.X, data.t, e)
    assert table["smd_unweighted"].abs().max() > 0.2
    assert table["smd_weighted"].abs().max() < 0.1
    assert table["balanced"].all()


def test_weight_definitions():
    e, t = np.array([0.2, 0.5, 0.8]), np.array([1, 0, 0])
    np.testing.assert_allclose(ate_weights(e, t), [5.0, 2.0, 5.0])
    np.testing.assert_allclose(att_weights(e, t), [1.0, 1.0, 4.0])
