import pytest

from loyalty_spend.models.tuning import nested_cv, search_space, tune

pytest.importorskip("optuna")


def test_search_space_unknown_model():
    import optuna

    with pytest.raises(KeyError):
        search_space(optuna.trial.FixedTrial({}), "ols")


def test_tune_returns_buildable_params(xy):
    X, y = xy
    params, score = tune("hist_gbm", X, y, n_trials=3, n_splits=3)
    assert {"learning_rate", "max_iter"} <= set(params)
    assert score > 0


@pytest.mark.slow
def test_nested_cv_reports_optimism(xy):
    X, y = xy
    res = nested_cv("hist_gbm", X, y, outer_splits=2, inner_splits=2, n_trials=2)
    assert len(res.outer_rmse) == 2 and len(res.params) == 2
    assert isinstance(res.optimism, float)
