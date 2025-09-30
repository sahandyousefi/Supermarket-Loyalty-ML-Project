import numpy as np
import pytest

from loyalty_spend.segments.gower import gower_matrix
from loyalty_spend.synthetic.conditional import ConditionalCopulaSynthesizer
from loyalty_spend.synthetic.privacy import (
    gower_cross,
    nearest_distances,
    privacy_report,
)


@pytest.fixture(scope="module")
def parts(train_df):
    data = train_df.drop(columns="customer_id").sample(frac=1.0, random_state=0)
    return data.iloc[:600].reset_index(drop=True), data.iloc[600:].reset_index(
        drop=True
    )


def test_cross_distance_matches_gower_matrix_on_same_rows(parts):
    train, _ = parts
    rows = train.iloc[:50]
    expected = gower_matrix(
        rows, numeric=["spend", "first_month", "items_in_first_month"]
    )
    assert np.allclose(gower_cross(rows, rows), expected, atol=1e-12)


def test_copies_are_detected(parts):
    train, holdout = parts
    copies = train.iloc[:100].reset_index(drop=True)
    report = privacy_report(train, holdout, copies)
    assert report["exact_copies"] == 1.0
    assert report["share_closer_than_holdout_q05"] == 1.0


def test_nndr_is_between_zero_and_one(parts):
    train, holdout = parts
    out = nearest_distances(holdout, train)
    assert out["nndr"].between(0, 1).all()
    assert (out["dcr"] > 0).all()


def test_conditional_synthesiser_does_not_copy_members(parts):
    train, holdout = parts
    synth = ConditionalCopulaSynthesizer("loyalty_years", seed=0).fit(train)
    report = privacy_report(train, holdout, synth.sample(len(holdout)))
    assert report["exact_copies"] == 0.0
    assert report["share_closer_than_holdout_q05"] < 0.1
