import numpy as np
import pytest

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.doubly_robust import aipw_scores, crossfit_nuisance
from loyalty_spend.causal.heterogeneity import blp_test, clan, crossfit_cate, gates
from loyalty_spend.causal.metalearners import TLearner
from loyalty_spend.causal.simulate import simulate_promotion


def _blp(train_df, heterogeneity, seed):
    study = simulate_promotion(
        train_df, n=3000, effect=5.0, heterogeneity=heterogeneity, seed=seed
    )
    data = from_frame(study.frame)
    proxy = crossfit_cate(TLearner, data.X, data.t, data.y, n_folds=3)
    psi = aipw_scores(data, crossfit_nuisance(data))
    return study, blp_test(proxy, psi)


def test_blp_detects_real_heterogeneity(train_df):
    study, out = _blp(train_df, heterogeneity=3.0, seed=22)
    assert out["ate"] == pytest.approx(study.true_ate, abs=3 * out["ate_se"])
    assert out["heterogeneity"] == pytest.approx(1.0, abs=0.3)
    assert out["heterogeneity_p_value"] < 1e-6


def test_blp_finds_none_when_the_effect_is_constant(train_df):
    _, out = _blp(train_df, heterogeneity=0.0, seed=23)
    assert out["heterogeneity_p_value"] > 0.01


def test_hc1_errors_match_the_homoscedastic_case_roughly():
    rng = np.random.default_rng(0)
    s = rng.normal(size=5000)
    psi = 2.0 + 0.5 * s + rng.normal(size=5000)
    out = blp_test(s, psi)
    assert out["heterogeneity_se"] == pytest.approx(1 / np.sqrt(5000), rel=0.1)


def test_gates_and_clan_point_to_long_tenure(train_df):
    study = simulate_promotion(train_df, n=3000, effect=5.0, heterogeneity=3.0, seed=24)
    data = from_frame(study.frame)
    proxy = crossfit_cate(TLearner, data.X, data.t, data.y, n_folds=3)
    psi = aipw_scores(data, crossfit_nuisance(data))
    out = gates(proxy, psi, n_groups=5)
    assert out["top_minus_bottom"] > 6.0
    assert out["p_value"] < 1e-6
    profile = clan(data.X, proxy, n_groups=5).set_index("covariate")
    assert profile.loc["loyalty_years=10+", "difference"] > 0.5
    assert profile.loc["loyalty_years=0-1", "difference"] < -0.5
