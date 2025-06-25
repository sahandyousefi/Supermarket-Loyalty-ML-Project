import numpy as np
import pytest

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.metalearners import SLearner, TLearner, XLearner, pehe
from loyalty_spend.causal.simulate import simulate_promotion


@pytest.fixture(scope="module")
def hetero(train_df):
    study = simulate_promotion(
        train_df, n=4000, effect=5.0, heterogeneity=3.0, confounding=1.0, seed=20
    )
    return study, from_frame(study.frame)


def test_pehe():
    assert pehe([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert pehe([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_s_learner_finds_the_average_and_some_heterogeneity(hetero):
    study, data = hetero
    tau_hat = SLearner().fit(data.X, data.t, data.y).effect(data.X)
    assert tau_hat.mean() == pytest.approx(study.true_ate, abs=1.0)
    assert np.corrcoef(tau_hat, study.effect)[0, 1] > 0.5
    assert pehe(tau_hat, study.effect) < np.std(study.effect)


def test_t_learner_recovers_the_tenure_gradient(hetero):
    study, data = hetero
    tau_hat = TLearner().fit(data.X, data.t, data.y).effect(data.X)
    bands = data.X["loyalty_years"].astype(str).to_numpy()
    by_band = {b: tau_hat[bands == b].mean() for b in ("0-1", "10+")}
    assert by_band["10+"] - by_band["0-1"] == pytest.approx(12.0, abs=3.0)
    assert pehe(tau_hat, study.effect) < np.std(study.effect)


def test_x_learner_helps_when_one_arm_is_small(train_df):
    study = simulate_promotion(
        train_df, n=3000, effect=5.0, heterogeneity=3.0, treated_share=0.1, seed=21
    )
    data = from_frame(study.frame)
    t_err = pehe(TLearner().fit(data.X, data.t, data.y).effect(data.X), study.effect)
    x_err = pehe(XLearner().fit(data.X, data.t, data.y).effect(data.X), study.effect)
    assert x_err < t_err
