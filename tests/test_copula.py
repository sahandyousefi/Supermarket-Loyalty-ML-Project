import numpy as np
import pandas as pd
import pytest

from loyalty_spend.synthetic.copula import (
    GaussianCopulaSynthesizer,
    nearest_correlation,
)


def test_nearest_correlation_is_valid():
    r = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]])
    fixed = nearest_correlation(r)
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.linalg.eigvalsh(fixed).min() > 0


def test_sample_keeps_schema_and_marginals(train_df):
    data = train_df.drop(columns="customer_id")
    synth = GaussianCopulaSynthesizer(seed=0).fit(data).sample(4000)
    assert list(synth.columns) == list(data.columns)
    for col in ("region", "loyalty_years"):
        assert list(synth[col].cat.categories) == list(data[col].cat.categories)
        real = data[col].value_counts(normalize=True)
        fake = synth[col].value_counts(normalize=True)
        assert (real - fake).abs().max() < 0.03
    assert synth["spend"].between(data["spend"].min(), data["spend"].max()).all()
    assert (synth["items_in_first_month"] % 1 == 0).all()
    assert abs(synth["spend"].median() - data["spend"].median()) < 2.0


def test_rank_correlations_of_amounts_are_kept(train_df):
    data = train_df.drop(columns="customer_id")
    synth = GaussianCopulaSynthesizer(seed=1).fit(data).sample(4000)
    cols = ["spend", "first_month", "items_in_first_month"]
    gap = data[cols].corr("spearman") - synth[cols].corr("spearman")
    assert gap.abs().to_numpy().max() < 0.08


def test_missingness_is_reproduced():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    frame = pd.DataFrame(
        {
            "a": np.where(x > 1, np.nan, x),
            "b": x + rng.normal(scale=0.1, size=2000),
            "c": pd.Categorical(rng.choice(["p", "q"], size=2000)),
        }
    )
    synth = GaussianCopulaSynthesizer(seed=2).fit(frame).sample(5000)
    share = synth["a"].isna().mean()
    assert share == pytest.approx(frame["a"].isna().mean(), abs=0.02)
    # Missing values occur where b is large, as in the real data.
    b = synth["b"].groupby(synth["a"].isna()).mean()
    assert b[True] - b[False] > 0.5


def test_sample_before_fit_raises():
    with pytest.raises(RuntimeError):
        GaussianCopulaSynthesizer().sample(5)
