import pytest

from loyalty_spend.data.contracts import load_contracts
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.persistence import ModelBundle
from loyalty_spend.models.registry import build_model
from loyalty_spend.ops.monitor import monitor_batch


@pytest.fixture(scope="module")
def bundle(train_df):
    X, y = split_xy(train_df)
    return ModelBundle("ols", build_model("ols").fit(X, y), y.to_numpy() * 0)


def test_same_population_is_ok(train_df, bundle):
    result = monitor_batch(train_df.iloc[:500], train_df.iloc[500:], bundle)
    assert result.status == "ok", result.reasons
    assert set(result.to_dict()) >= {"status", "drift", "prediction_shift"}


def test_shifted_batch_raises_an_alert(train_df, bundle):
    batch = train_df.iloc[500:].copy()
    batch["first_month"] = batch["first_month"] * 1.8
    result = monitor_batch(train_df.iloc[:500], batch, bundle)
    assert result.status == "alert"
    assert any("first_month" in r for r in result.reasons)


def test_contract_errors_raise_an_alert(train_df, bundle):
    batch = train_df.iloc[500:].copy()
    batch["customer_id"] = 7
    result = monitor_batch(
        train_df.iloc[:500], batch, bundle, contract=load_contracts()["training"]
    )
    assert result.status == "alert"
    assert not result.contract_violations.empty
