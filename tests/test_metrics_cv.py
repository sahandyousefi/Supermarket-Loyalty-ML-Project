import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from loyalty_spend.evaluation.cv import (
    compare_results,
    corrected_ttest,
    cross_validate,
    make_splits,
)
from loyalty_spend.evaluation.metrics import mape, regression_report, rmse
from loyalty_spend.models.registry import build_model


def test_rmse_matches_sklearn():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))


def test_mape_rejects_zero_targets():
    with pytest.raises(ValueError):
        mape([0.0, 1.0], [1.0, 1.0])


def test_report_keys():
    assert set(regression_report([1.0, 2.0], [1.0, 2.5])) == {
        "rmse",
        "mae",
        "mape",
        "r2",
    }


def test_splits_cover_every_row_once_per_repeat():
    splits = make_splits(50, n_splits=5, n_repeats=2, seed=0)
    assert len(splits) == 10
    counts = np.bincount(np.concatenate([te for _, te in splits]), minlength=50)
    assert (counts == 2).all()


def test_corrected_ttest_is_more_conservative_than_naive():
    d = np.random.default_rng(0).normal(0.05, 0.05, 15)
    stat, p = corrected_ttest(d, 800, 200)
    naive = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(stat) < abs(naive) and 0 <= p <= 1
    assert corrected_ttest(np.zeros(4), 80, 20) == (0.0, 1.0)


def test_cross_validate_and_compare(xy):
    X, y = xy
    splits = make_splits(len(y), 3, 1, 0)
    a = cross_validate("ols", build_model("ols"), X, y, splits)
    b = cross_validate("hist_gbm", build_model("hist_gbm", max_iter=50), X, y, splits)
    assert len(a.folds) == 3 and a.oof.shape == (len(y),)
    table = compare_results([a, b], reference="ols")
    assert set(table["model"]) == {"ols", "hist_gbm"}
    assert np.isnan(table.set_index("model").loc["ols", "p_value"])
    with pytest.raises(KeyError):
        compare_results([a], reference="ridge")


def test_cross_validate_does_not_mutate_estimator(xy):
    X, y = xy
    est = LinearRegression()
    cross_validate(
        "lr", est, X.select_dtypes("number"), y, make_splits(len(y), 3, 1, 0)
    )
    assert not hasattr(est, "coef_")
