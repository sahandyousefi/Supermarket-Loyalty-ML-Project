import numpy as np
import pytest

from loyalty_spend.data.contracts import load_contracts
from loyalty_spend.data.io import load_clean, load_modeling_data
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.persistence import ModelBundle
from loyalty_spend.models.registry import build_model
from loyalty_spend.ops.batch import score_batch


@pytest.fixture(scope="module")
def bundle_and_test(raw_dir):
    train, _ = load_modeling_data(raw_dir / "train.csv", raw_dir / "test.csv")
    X, y = split_xy(train)
    model = build_model("ols").fit(X.iloc[:700], y.iloc[:700])
    resid = y.iloc[700:].to_numpy() - model.predict(X.iloc[700:])
    test, _ = load_clean(raw_dir / "test.csv", impute=False, require_target=False)
    return ModelBundle("ols", model, resid), test, train


def test_scores_with_intervals(bundle_and_test):
    bundle, test, _ = bundle_and_test
    result = score_batch(bundle, test, alpha=0.1, contract=load_contracts()["scoring"])
    s = result.scores
    assert len(s) == len(test)
    assert (s["lower"] <= s["prediction"]).all() and (
        s["prediction"] <= s["upper"]
    ).all()
    assert result.contract.ok
    widths = (s["upper"] - s["prediction"]).round(9).unique()
    assert len(widths) == 1


def test_strict_mode_stops_on_a_broken_batch(bundle_and_test):
    bundle, test, _ = bundle_and_test
    broken = test.copy()
    broken["customer_id"] = 1
    with pytest.raises(ValueError, match="contract violated"):
        score_batch(bundle, broken, contract=load_contracts()["scoring"])
    lenient = score_batch(
        bundle, broken, contract=load_contracts()["scoring"], strict=False
    )
    assert not lenient.contract.ok and len(lenient.scores) == len(test)


def test_interval_coverage_on_held_out_training_rows(bundle_and_test):
    bundle, _, train = bundle_and_test
    held = train.iloc[850:]
    s = score_batch(bundle, held, alpha=0.1).scores
    covered = (held["spend"].to_numpy() >= s["lower"]) & (
        held["spend"].to_numpy() <= s["upper"]
    )
    assert np.mean(covered) > 0.8
