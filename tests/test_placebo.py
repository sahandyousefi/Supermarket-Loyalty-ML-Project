import numpy as np

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.placebo import placebo_test
from loyalty_spend.causal.simulate import simulate_promotion


def test_real_effect_stands_out_from_placebo_draws(train_df):
    data = from_frame(simulate_promotion(train_df, n=2000, effect=6.0, seed=13).frame)
    result = placebo_test(data, n_permutations=100)
    assert result.p_value <= 1 / 101 + 1e-12
    assert abs(np.mean(result.placebo)) < 3 * result.placebo_sd / np.sqrt(100) + 0.5


def test_no_effect_gives_a_large_p_value(train_df):
    data = from_frame(simulate_promotion(train_df, n=2000, effect=0.0, seed=14).frame)
    result = placebo_test(data, n_permutations=100, strata="loyalty_years")
    # Under no effect the p-value is uniform; 0.01 keeps false alarms rare.
    assert result.p_value > 0.01
    assert 0 < result.p_value <= 1
