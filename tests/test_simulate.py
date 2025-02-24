import numpy as np
import pytest

from loyalty_spend.data.schema import COLUMNS
from loyalty_spend.data.simulate import SCENARIOS, Scenario, simulate
from loyalty_spend.evaluation.simulation_study import run_study, winners


def test_simulated_frame_has_schema(train_df):
    data = simulate(train_df, SCENARIOS["linear"], n=300, seed=0)
    assert list(data.frame.columns) == COLUMNS
    assert (data.frame["spend"] > 0).all()
    assert data.frame["customer_id"].is_unique


def test_noise_level_matches_specification(train_df):
    data = simulate(train_df, Scenario("x", noise=0.1), n=20000, seed=0)
    resid = data.frame["spend"] - data.signal
    assert resid.std() == pytest.approx(data.bayes_rmse, rel=0.05)


def test_heteroscedastic_noise_grows_with_signal(train_df):
    data = simulate(train_df, SCENARIOS["noisy"], n=2000, seed=0)
    assert np.corrcoef(data.signal, data.noise_sd)[0, 1] > 0.99


def test_reproducible(train_df):
    a = simulate(train_df, SCENARIOS["interactions"], n=100, seed=5).frame
    b = simulate(train_df, SCENARIOS["interactions"], n=100, seed=5).frame
    assert a.equals(b)


@pytest.mark.slow
def test_study_ranks_linear_first_in_linear_scenario(train_df):
    table = run_study(
        train_df,
        ["ols", "hist_gbm"],
        scenarios=["linear", "interactions"],
        n=600,
        n_splits=3,
    )
    best = winners(table).set_index("scenario")["model"]
    assert best["linear"] == "ols"
    assert best["interactions"] == "hist_gbm"
    assert (table["rmse"] >= 0.9 * table["bayes_rmse"]).all()
