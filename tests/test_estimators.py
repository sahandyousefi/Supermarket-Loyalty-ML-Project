import numpy as np
import pytest

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.estimators import (
    EffectEstimate,
    hajek_ipw,
    ipw,
    regression_adjustment,
)
from loyalty_spend.causal.simulate import simulate_promotion


@pytest.fixture(scope="module")
def confounded(train_df):
    study = simulate_promotion(train_df, n=4000, effect=6.0, confounding=1.5, seed=5)
    return study, from_frame(study.frame)


def test_hajek_ipw_with_the_true_propensity_is_unbiased(confounded):
    study, data = confounded
    est = hajek_ipw(data.y, data.t, study.propensity)
    naive_bias = abs(data.naive_difference() - study.true_ate)
    assert naive_bias > 5.0
    # Weighting is noisy under strong confounding, but removes most of the bias.
    assert abs(est - study.true_ate) < 0.4 * naive_bias


@pytest.mark.parametrize("estimand", ["ate", "att"])
def test_ipw_interval_covers_the_truth(confounded, estimand):
    study, data = confounded
    result = ipw(data, estimand, n_boot=200)
    truth = study.true_ate if estimand == "ate" else study.true_att
    assert result.ci_lower < truth < result.ci_upper
    assert result.se > 0
    assert result.to_dict()["p_value"] < 0.05


def test_estimate_record_and_invalid_estimand(confounded):
    _, data = confounded
    rec = EffectEstimate("x", "ate", 1.0, 0.0, 1.0, 1.0, 10)
    assert np.isnan(rec.p_value)
    with pytest.raises(ValueError):
        ipw(data, "atc")


@pytest.mark.parametrize("estimand", ["ate", "att"])
def test_regression_adjustment_recovers_the_effect(confounded, estimand):
    study, data = confounded
    result = regression_adjustment(data, estimand, n_boot=100)
    truth = study.true_ate if estimand == "ate" else study.true_att
    assert result.ci_lower < truth < result.ci_upper
    # A correct linear outcome model is far more precise than weighting.
    assert result.se < ipw(data, estimand, n_boot=100).se
