import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from loyalty_spend.models.persistence import (  # noqa: E402
    ModelBundle,
    data_fingerprint,
    save_bundle,
)
from loyalty_spend.models.registry import build_model  # noqa: E402
from loyalty_spend.serving.api import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory, train_df):
    from loyalty_spend.features.preprocess import split_xy

    X, y = split_xy(train_df)
    model = build_model("ols").fit(X, y)
    bundle = ModelBundle(
        "ols",
        model,
        np.random.default_rng(0).normal(scale=0.3, size=200),
        metadata={"data_fingerprint": data_fingerprint(X)},
    )
    path = save_bundle(bundle, tmp_path_factory.mktemp("m") / "model.joblib")
    return TestClient(create_app(path))


def test_health_and_version(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/version").json()
    assert info["model"] == "ols"
    assert len(info["data_fingerprint"]) == 12


CUSTOMER = {
    "customer_id": 1,
    "first_month": 12.5,
    "items_in_first_month": 6,
    "region": "Europe",
    "loyalty_years": "3-5",
    "joining_month": "Mar",
    "promotion": "No",
}


def test_predict_matches_the_bundle(client, train_df):
    from loyalty_spend.features.preprocess import feature_frame

    body = {
        "customers": [CUSTOMER, {**CUSTOMER, "customer_id": 2, "first_month": None}]
    }
    out = client.post("/predict", json=body).json()
    assert out["coverage"] == pytest.approx(0.9)
    first, second = out["predictions"]
    assert first["lower"] <= first["spend"] <= first["upper"]
    assert second["customer_id"] == 2
    row = train_df.iloc[[0]].copy()
    for k, v in CUSTOMER.items():
        if k != "customer_id":
            row[k] = v
    bundle = client.app.state.bundle
    expected = bundle.predict(
        feature_frame(row.astype({c: train_df[c].dtype for c in row.columns}))
    )[0]
    assert first["spend"] == pytest.approx(expected, abs=0.01)
