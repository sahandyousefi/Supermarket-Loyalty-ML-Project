import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from loyalty_spend.evaluation.cv import (
    corrected_ttest,
    make_splits,
)
from loyalty_spend.evaluation.metrics import mape, regression_report, rmse


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
