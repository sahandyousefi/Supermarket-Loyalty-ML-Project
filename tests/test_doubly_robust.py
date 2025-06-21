import numpy as np
import pytest
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.doubly_robust import (
    aipw,
    aipw_att,
    aipw_scores,
    crossfit_nuisance,
    frame_of,
)
from loyalty_spend.causal.simulate import simulate_promotion


@pytest.fixture(scope="module")
def confounded(train_df):
    study = simulate_promotion(train_df, n=4000, effect=6.0, confounding=1.5, seed=6)
    return study, from_frame(study.frame)


def test_aipw_covers_the_true_ate(confounded):
    study, data = confounded
    result = aipw(data)
    assert result.ci_lower < study.true_ate < result.ci_upper
    assert result.se < 1.0


def test_aipw_survives_a_wrong_propensity_model(confounded):
    study, data = confounded
    # A propensity model that sees only the region is misspecified, but the
    # outcome models are correct, so the estimate stays consistent.
    region_only = make_pipeline(
        make_column_transformer((OneHotEncoder(), ["region"])),
        LogisticRegression(max_iter=1000),
    )
    nuisance = crossfit_nuisance(data, propensity=region_only)
    result = aipw(data, nuisance)
    assert result.ci_lower < study.true_ate < result.ci_upper


def test_scores_reduce_to_the_difference_of_predictions_without_residuals(confounded):
    _, data = confounded
    nuisance = crossfit_nuisance(data)
    fitted = data.__class__(
        data.X, data.t, np.where(data.t == 1, nuisance.mu1, nuisance.mu0)
    )
    np.testing.assert_allclose(
        aipw_scores(fitted, nuisance), nuisance.mu1 - nuisance.mu0
    )


def test_aipw_att_covers_the_effect_on_the_treated(train_df):
    # The effect varies with tenure and uptake depends on tenure, so the
    # effect on the treated is a different quantity from the average effect.
    study = simulate_promotion(
        train_df, n=4000, effect=6.0, confounding=1.5, heterogeneity=2.0, seed=7
    )
    assert study.true_att != pytest.approx(study.true_ate, abs=1e-3)
    data = from_frame(study.frame)
    result = aipw_att(data)
    assert result.estimand == "att"
    assert result.ci_lower < study.true_att < result.ci_upper


def test_frame_of_estimates(confounded):
    _, data = confounded
    table = frame_of([aipw(data), aipw_att(data)])
    assert list(table["estimand"]) == ["ate", "att"]
    assert {"estimate", "se", "p_value"} <= set(table.columns)
