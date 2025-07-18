import numpy as np
import pytest

from loyalty_spend.business.breakdown import cost_by_group
from loyalty_spend.business.rewards import RewardSchedule, forecast_liability

SCHEDULE = RewardSchedule(((0.0, 0.0), (100.0, 0.02), (125.0, 0.03), (140.0, 0.05)))


def test_cost_by_region_on_the_scoring_file(raw_dir):
    from loyalty_spend.data.io import load_modeling_data
    from loyalty_spend.features.preprocess import feature_frame, split_xy
    from loyalty_spend.models.registry import build_model

    train, test = load_modeling_data(raw_dir / "train.csv", raw_dir / "test.csv")
    model = build_model("ols").fit(*split_xy(train))
    pred = model.predict(feature_frame(test))
    fc = forecast_liability(
        test["customer_id"],
        pred,
        np.random.default_rng(0).normal(scale=0.3, size=500),
        SCHEDULE,
        300,
        0,
    )
    table = cost_by_group(fc, test["region"], name="region")
    assert set(table["region"]) <= {
        "Americas",
        "Asia/Pacific",
        "Europe",
        "Middle East/Africa",
        "Unknown",
    }
    assert table["customers"].sum() == len(test)
    assert table["share_of_expected_cost"].sum() == pytest.approx(1.0)
