import numpy as np
import pytest

from loyalty_spend.business.rewards import RewardSchedule, forecast_liability
from loyalty_spend.segments.value import liability_by_segment

SCHEDULE = RewardSchedule(((0.0, 0.0), (100.0, 0.02), (125.0, 0.03), (140.0, 0.05)))


def test_segment_totals_add_up():
    rng = np.random.default_rng(0)
    pred = rng.uniform(90, 150, size=300)
    fc = forecast_liability(
        range(300), pred, rng.normal(scale=4, size=500), SCHEDULE, 500, 1
    )
    labels = (pred > 125).astype(int)
    table = liability_by_segment(fc, labels).set_index("segment")
    assert table["customers"].sum() == 300
    assert table["expected_reward"].sum() == pytest.approx(
        fc.per_customer["expected_reward"].sum()
    )
    assert table["share_of_expected_cost"].sum() == pytest.approx(1.0)
    assert table.loc[1, "mean_predicted_spend"] > table.loc[0, "mean_predicted_spend"]
