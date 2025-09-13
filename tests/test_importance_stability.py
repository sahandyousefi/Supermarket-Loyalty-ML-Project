import numpy as np
import pytest

from loyalty_spend.evaluation.cv import make_splits
from loyalty_spend.explain.stability import importance_stability, kendalls_w
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model


def test_kendalls_w_extremes():
    same = np.tile(np.arange(1, 6), (4, 1)).astype(float)
    assert kendalls_w(same) == pytest.approx(1.0)
    opposite = np.vstack([np.arange(1, 5), np.arange(4, 0, -1)]).astype(float)
    assert kendalls_w(opposite) == pytest.approx(0.0)


def test_importance_ranking_is_stable_on_the_supplied_data(train_df):
    X, y = split_xy(train_df)
    table, w = importance_stability(
        build_model("ols"), X, y, make_splits(len(y), 4, 1, 0), n_repeats=3
    )
    assert set(table["fold"]) == {0, 1, 2, 3}
    top = table.groupby("feature")["rmse_increase"].mean().idxmax()
    assert top == "loyalty_years"
    assert w > 0.8
