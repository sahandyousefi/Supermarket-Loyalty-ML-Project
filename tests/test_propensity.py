import numpy as np
import pytest

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.propensity import crossfit_propensity, propensity_model
from loyalty_spend.causal.simulate import simulate_promotion


@pytest.fixture(scope="module")
def study(train_df):
    return simulate_promotion(train_df, n=3000, confounding=1.5, seed=3)


@pytest.mark.parametrize("kind", ["logistic", "gbm"])
def test_crossfit_propensity_recovers_the_true_score(study, kind):
    data = from_frame(study.frame)
    model = propensity_model(data.X, kind)
    e = crossfit_propensity(data.X, data.t, model, seed=0)
    assert np.corrcoef(e, study.propensity)[0, 1] > 0.9
    assert e.mean() == pytest.approx(data.t.mean(), abs=0.02)
    brier = np.mean((e - data.t) ** 2)
    assert brier < np.mean((data.t.mean() - data.t) ** 2)


def test_clipping_and_unknown_kind(study):
    data = from_frame(study.frame)
    e = crossfit_propensity(data.X, data.t, clip=0.2)
    assert e.min() >= 0.2 and e.max() <= 0.8
    with pytest.raises(ValueError):
        propensity_model(data.X, "forest")
