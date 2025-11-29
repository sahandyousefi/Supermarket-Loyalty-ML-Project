import numpy as np
import pytest

from loyalty_spend.business.noise import compare_noise_models
from loyalty_spend.business.rewards import RewardSchedule

TIERS = ((0.0, 0.0), (100.0, 0.02), (125.0, 0.03), (140.0, 0.05))


@pytest.fixture(scope="module")
def result(train_df, raw_dir):
    from loyalty_spend.data.io import load_clean

    test, _ = load_clean(raw_dir / "test.csv", impute=False, require_target=False)
    return compare_noise_models(train_df, test, RewardSchedule(TIERS), 800, 0)


def test_every_noise_model_gives_a_forecast(result):
    table, forecasts = result
    assert len(table) == 3 and set(forecasts) == set(table["noise model"])
    assert (table["p05"] <= table["expected_total"]).all()
    assert (table["expected_total"] <= table["p95"]).all()


def test_means_agree_while_spreads_differ(result):
    table, _ = result
    totals = table["expected_total"].to_numpy()
    assert np.ptp(totals) < 0.01 * totals.mean()
    spread = (table["p95"] - table["p05"]).to_numpy()
    assert spread.max() > spread.min()
