import numpy as np

from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model
from loyalty_spend.robustness.stress import stress_test


def test_damage_raises_error_and_more_damage_raises_it_more(train_df):
    X, y = split_xy(train_df)
    model = build_model("ols").fit(X.iloc[:700], y.iloc[:700])
    table = stress_test(
        model.predict, X.iloc[700:].reset_index(drop=True), y.iloc[700:]
    )
    assert np.isfinite(table["rmse"]).all()
    assert (table["rmse_ratio"] >= 0.99).all()
    for _, part in table.groupby("scenario"):
        assert (
            part["rmse"].is_monotonic_increasing
            or part["rmse"].max() / part["rmse"].min() < 1.05
        )
    worst = table.sort_values("rmse_ratio").iloc[-1]
    assert worst["scenario"] in {"missing categories", "unseen categories"}
