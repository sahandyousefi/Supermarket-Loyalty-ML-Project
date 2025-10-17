import numpy as np
import pandas as pd
import pytest

from loyalty_spend.experiment.adjustment import difference_in_means
from loyalty_spend.experiment.randomisation import (
    standardised_differences,
    stratified_assignment,
    stratified_effect,
)


def test_arms_are_balanced_within_every_stratum(train_df):
    strata = train_df[["loyalty_years", "region"]]
    t = stratified_assignment(strata, 0.5, seed=1)
    counts = pd.crosstab([strata["loyalty_years"], strata["region"]], t)
    assert ((counts[True] - counts[False]).abs() <= 1).all()
    smd = standardised_differences(strata, t)
    assert smd.abs().max() < 0.03


def test_unequal_share_is_respected_on_average():
    strata = pd.Series(np.repeat(["a", "b", "c"], [7, 11, 13]))
    shares = [stratified_assignment(strata, 0.3, seed=s).mean() for s in range(400)]
    assert np.mean(shares) == pytest.approx(0.3, abs=0.01)


def test_stratification_reduces_the_variance_of_the_estimate():
    rng = np.random.default_rng(0)
    band = pd.Series(rng.choice(["low", "mid", "high"], 300))
    base = band.map({"low": 70.0, "mid": 140.0, "high": 130.0}).to_numpy()
    plain, strat = [], []
    for s in range(300):
        y = base + rng.normal(0, 3, 300)
        t_simple = rng.permutation(np.arange(300) < 150)
        plain.append(difference_in_means(y + 1.0 * t_simple, t_simple).estimate)
        t = stratified_assignment(band, 0.5, seed=s)
        strat.append(stratified_effect(y + 1.0 * t, t, band).estimate)
    assert np.mean(strat) == pytest.approx(1.0, abs=0.1)
    assert np.std(strat) < np.std(plain) / 5
