from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.dml import dml_plr
from loyalty_spend.causal.simulate import simulate_promotion


def test_dml_recovers_a_constant_effect(train_df):
    study = simulate_promotion(train_df, n=4000, effect=6.0, confounding=1.5, seed=8)
    data = from_frame(study.frame)
    result = dml_plr(data, n_repeats=2)
    assert result.ci_lower < 6.0 < result.ci_upper
    assert abs(result.estimate - 6.0) < 0.3 * abs(data.naive_difference() - 6.0)


def test_dml_finds_no_effect_when_there_is_none(train_df):
    study = simulate_promotion(train_df, n=3000, effect=0.0, confounding=1.0, seed=9)
    result = dml_plr(from_frame(study.frame), n_repeats=1)
    assert result.ci_lower < 0.0 < result.ci_upper
    assert result.p_value > 0.01
