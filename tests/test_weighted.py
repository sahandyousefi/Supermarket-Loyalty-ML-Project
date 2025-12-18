import numpy as np
import pandas as pd
import pytest

from loyalty_spend.models.registry import available_models, build_model
from loyalty_spend.models.weighted import GroupWeightedRegression


def _data(rng, n=2000):
    month = rng.choice(["Jan", "Aug"], n, p=[0.8, 0.2])
    X = pd.DataFrame(
        {
            "first_month": rng.uniform(10, 50, n),
            "items_in_first_month": rng.integers(5, 40, n).astype(float),
            "region": pd.Categorical(rng.choice(["Americas", "Europe"], n)),
            "loyalty_years": pd.Categorical(rng.choice(["0-1", "1-3"], n)),
            "joining_month": pd.Categorical(month),
            "promotion": pd.Categorical(rng.choice(["Yes", "No"], n)),
        }
    )
    sd = np.where(month == "Aug", 5.0, 0.5)
    y = 100 + 2.0 * X["first_month"].to_numpy() + rng.normal(0, sd)
    return X, y, sd


def test_group_variances_are_recovered():
    X, y, _ = _data(np.random.default_rng(0))
    model = GroupWeightedRegression(shrinkage=0.0).fit(X, y)
    assert np.sqrt(model.variance_["Jan"]) == pytest.approx(0.5, rel=0.1)
    assert np.sqrt(model.variance_["Aug"]) == pytest.approx(5.0, rel=0.1)


def test_weighting_is_more_efficient_than_ols():
    rng = np.random.default_rng(1)
    X_new, _, _ = _data(np.random.default_rng(99), 500)
    truth = 100 + 2.0 * X_new["first_month"].to_numpy()
    errors = {"ols": [], "wls_month": []}
    for _ in range(40):
        X, y, _ = _data(rng, 400)
        for name in errors:
            pred = build_model(name).fit(X, y).predict(X_new)
            errors[name].append(np.mean((pred - truth) ** 2))
    assert np.mean(errors["wls_month"]) < 0.7 * np.mean(errors["ols"])


def test_unseen_group_gets_the_pooled_variance():
    X, y, _ = _data(np.random.default_rng(2), 500)
    model = build_model("wls_month").fit(X, y)
    new = X.iloc[:3].copy()
    new["joining_month"] = pd.Categorical(["Dec"] * 3)
    assert np.allclose(model.predict_sd(new), np.sqrt(model.pooled_variance_))
    assert "wls_month" in available_models()


def test_weighted_intervals_are_calibrated_within_groups():
    from loyalty_spend.models.weighted import (
        cross_fitted_predictive,
        predictive_comparison,
    )

    X, y, _ = _data(np.random.default_rng(4), 1500)
    table = cross_fitted_predictive(X, y, n_splits=3)
    result = predictive_comparison(table, X["joining_month"]).set_index("model")
    assert result.loc["ols", "months_flagged"] == 2
    assert result.loc["wls", "months_flagged"] == 0
    assert result.loc["wls", "crps"] < 0.8 * result.loc["ols", "crps"]


def test_standardized_residuals_have_unit_scale_in_every_group():
    from loyalty_spend.models.weighted import standardized_residuals

    X, y, _ = _data(np.random.default_rng(6), 3000)
    model = GroupWeightedRegression(shrinkage=0.0).fit(X, y)
    r = standardized_residuals(model, X, y)
    for month in ("Jan", "Aug"):
        sel = (X["joining_month"] == month).to_numpy()
        assert np.sqrt(np.mean(r[sel] ** 2)) == pytest.approx(1.0, rel=0.05)


def test_weighted_bundle_scores_with_month_scaled_intervals(train_df, tmp_path):
    from loyalty_spend.evaluation.cv import make_splits
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.persistence import ModelBundle
    from loyalty_spend.models.weighted import cross_fitted_scaled_residuals
    from loyalty_spend.ops.batch import score_batch

    X, y = split_xy(train_df)
    splits = make_splits(len(y), 3, 1, 0)
    scaled = cross_fitted_scaled_residuals("wls_month", X, y, splits)
    model = build_model("wls_month").fit(X, y)
    bundle = ModelBundle(
        "wls_month",
        model,
        residuals=np.zeros(len(y)),
        metadata={"scaled_residuals": scaled.tolist()},
    )
    scores = score_batch(bundle, train_df, alpha=0.1).scores
    width = (scores["upper"] - scores["lower"]).to_numpy()
    month = train_df["joining_month"].astype(str).to_numpy()
    assert width[month == "Aug"].mean() > 3 * width[month == "Jun"].mean()
    covered = (train_df["spend"] >= scores["lower"]) & (
        train_df["spend"] <= scores["upper"]
    )
    assert covered.mean() == pytest.approx(0.9, abs=0.03)


def test_cohort_scale_recovers_a_power_law():
    from loyalty_spend.models.weighted import CohortScale

    rng = np.random.default_rng(7)
    spreads = np.linspace(0.1, 0.5, 12)
    months, first, resid = [], [], []
    for g, s in enumerate(spreads):
        months += [str(g)] * 300
        first += list(rng.normal(20, s, 300))
        resid += list(rng.normal(0, 1.2 * s, 300))
    scale = CohortScale().fit(months, first, resid)
    assert scale.slope_ == pytest.approx(1.0, abs=0.1)
    assert scale.r2_ > 0.95
    assert scale.predict(rng.normal(20, 0.3, 500)) == pytest.approx(0.36, rel=0.15)
