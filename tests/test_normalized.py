import numpy as np
import pandas as pd
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.scoring import interval_score
from loyalty_spend.models.registry import build_model
from loyalty_spend.uncertainty.conformal import SplitConformalRegressor
from loyalty_spend.uncertainty.normalized import NormalizedConformalRegressor


@pytest.fixture(scope="module")
def fitted(train_df):
    df = simulate(train_df, SCENARIOS["noisy"], n=6000, seed=31).frame
    X, y = df[FEATURES].iloc[:3000], df["spend"].iloc[:3000]
    X_new, y_new = df[FEATURES].iloc[3000:], df["spend"].iloc[3000:].to_numpy()
    split = SplitConformalRegressor(build_model("ols")).fit(X, y)
    normalized = NormalizedConformalRegressor(build_model("ols")).fit(X, y)
    return split, normalized, X_new, y_new


def test_marginal_coverage_holds(fitted):
    _, normalized, X_new, y_new = fitted
    assert normalized.predict_interval(X_new).coverage(y_new) == pytest.approx(
        0.9, abs=0.02
    )


def _quintile_coverage(iv, y, pred):
    covered = (y >= iv.lower) & (y <= iv.upper)
    return pd.Series(covered).groupby(pd.qcut(pred, 5, labels=False)).mean()


def test_intervals_adapt_to_the_noise_level(fitted):
    split, normalized, X_new, y_new = fitted
    iv_s, iv_n = split.predict_interval(X_new), normalized.predict_interval(X_new)
    pred = normalized.predict(X_new)
    top = pred > np.quantile(pred, 0.8)
    assert iv_n.width[top].mean() > iv_n.width[~top].mean()
    # Split conformal spreads one width over everyone and misses most for the
    # high spenders; the normalized intervals keep every quintile near 90%.
    cov_s = _quintile_coverage(iv_s, y_new, pred)
    cov_n = _quintile_coverage(iv_n, y_new, pred)
    assert (cov_n - 0.9).abs().max() < (cov_s - 0.9).abs().max()
    assert cov_n.min() > 0.85
    # Better conditional coverage does not cost a worse interval score.
    score_s = interval_score(y_new, iv_s.lower, iv_s.upper, 0.1).mean()
    score_n = interval_score(y_new, iv_n.lower, iv_n.upper, 0.1).mean()
    assert score_n < 1.03 * score_s
