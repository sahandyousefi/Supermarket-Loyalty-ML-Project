import pytest

from loyalty_spend.causal.analysis import analyse_promotion, naive_difference
from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.simulate import simulate_promotion


@pytest.fixture(scope="module")
def analysis(train_df):
    study = simulate_promotion(train_df, n=3000, effect=6.0, confounding=1.5, seed=15)
    return study, analyse_promotion(from_frame(study.frame), n_boot=100, n_repeats=1)


def test_every_estimator_is_reported(analysis):
    _, result = analysis
    table = result.estimates
    assert len(table) == 8
    assert set(table["estimand"]) == {"ate", "att"}
    assert (table["ci_lower"] < table["ci_upper"]).all()


def test_adjusted_estimators_beat_the_naive_difference(analysis):
    study, result = analysis
    table = result.estimates.set_index(["method", "estimand"])
    naive = table.loc[("Difference in means", "ate"), "estimate"]
    for method in ["Regression adjustment", "AIPW", "DML (partially linear)"]:
        est = table.loc[(method, "ate"), "estimate"]
        assert abs(est - study.true_ate) < abs(naive - study.true_ate)
    assert result.balance["balanced"].all()
    assert result.overlap["ess_treated"] > 0


def test_naive_difference_matches_the_study(train_df):
    data = from_frame(simulate_promotion(train_df, n=500, seed=16).frame)
    assert naive_difference(data).estimate == pytest.approx(data.naive_difference())
