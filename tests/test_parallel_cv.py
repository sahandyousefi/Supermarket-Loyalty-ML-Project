import numpy as np

from loyalty_spend.evaluation.cv import cross_validate, make_splits
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model


def test_parallel_folds_give_identical_results(train_df):
    X, y = split_xy(train_df)
    splits = make_splits(len(y), 3, 2, 0)
    for name in ("ols", "hist_gbm"):
        serial = cross_validate(name, build_model(name), X, y, splits)
        parallel = cross_validate(name, build_model(name), X, y, splits, n_jobs=2)
        assert np.array_equal(serial.oof, parallel.oof)
        assert serial.folds.equals(parallel.folds)


def test_fit_hook_runs_in_the_calling_process(train_df):
    X, y = split_xy(train_df)
    seen = []
    splits = make_splits(len(y), 3, 1, 0)
    cross_validate("ols", build_model("ols"), X, y, splits, seen.append, n_jobs=2)
    assert len(seen) == 3 and all(hasattr(m, "predict") for m in seen)
