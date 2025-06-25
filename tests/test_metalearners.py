import numpy as np
import pytest

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.metalearners import SLearner, pehe
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
