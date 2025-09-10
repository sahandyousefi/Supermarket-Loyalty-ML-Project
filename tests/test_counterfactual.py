import numpy as np
import pandas as pd
import pytest

from loyalty_spend.business.rewards import RewardSchedule
from loyalty_spend.explain.counterfactual import next_tier_counterfactual

SCHEDULE = RewardSchedule(((0.0, 0.0), (100.0, 0.02), (125.0, 0.03), (140.0, 0.05)))


def linear(Z):
    return 50 + 4 * Z["first_month"].to_numpy()


def test_exact_answer_for_a_linear_forecast():
    row = pd.DataFrame({"first_month": [20.0]})
    cf = next_tier_counterfactual(linear, row, SCHEDULE)
    assert cf.prediction == 130.0 and cf.next_threshold == 140.0
    assert cf.required_value == pytest.approx(22.5, abs=1e-3)
    assert cf.required_change == pytest.approx(2.5, abs=1e-3)


def test_infeasible_and_top_tier_cases():
    row = pd.DataFrame({"first_month": [20.0]})
    capped = next_tier_counterfactual(linear, row, SCHEDULE, upper=21.0)
    assert not capped.feasible and capped.required_change is None
    top = next_tier_counterfactual(
        linear, pd.DataFrame({"first_month": [30.0]}), SCHEDULE
    )
    assert top.next_threshold is None and top.required_change == 0.0
    with pytest.raises(ValueError):
        next_tier_counterfactual(
            linear, pd.DataFrame({"first_month": [1.0, 2.0]}), SCHEDULE
        )


def test_counterfactual_with_the_fitted_model(train_df):
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.registry import build_model

    X, y = split_xy(train_df)
    model = build_model("ols").fit(X, y)
    pred = model.predict(X)
    i = int(np.argmin(np.abs(pred - 120)))
    cf = next_tier_counterfactual(
        model.predict, X.iloc[[i]], SCHEDULE, upper=float(X["first_month"].max()) * 3
    )
    if cf.feasible:
        moved = model.predict(X.iloc[[i]].assign(first_month=cf.required_value))[0]
        assert moved == pytest.approx(cf.next_threshold, abs=0.01)
