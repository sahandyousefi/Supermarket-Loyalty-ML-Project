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


def test_reward_endpoint(client):
    customers = [
        {**CUSTOMER, "customer_id": i, "loyalty_years": y}
        for i, y in enumerate(["0-1", "5-10", "10+"])
    ]
    out = client.post(
        "/rewards", json={"customers": customers, "n_simulations": 500}
    ).json()
    assert (
        out["p05"]
        <= out["expected_total"]
        <= out["p95"]
        <= out["expected_shortfall_95"] + 1e-9
    )
    assert len(out["customers"]) == 3
    assert sum(c["expected_reward"] for c in out["customers"]) == pytest.approx(
        out["expected_total"], abs=0.05
    )


def test_invalid_requests_are_rejected(client):
    bad_level = {"customers": [{**CUSTOMER, "region": "Mars"}]}
    assert client.post("/predict", json=bad_level).status_code == 422
    negative = {"customers": [{**CUSTOMER, "first_month": -3}]}
    assert client.post("/predict", json=negative).status_code == 422
    assert client.post("/predict", json={"customers": []}).status_code == 422
    twice = {"customers": [CUSTOMER, CUSTOMER]}
    response = client.post("/predict", json=twice)
    assert response.status_code == 422 and "duplicated" in response.json()["detail"]
    assert (
        client.post(
            "/predict", json={"customers": [CUSTOMER], "alpha": 1.5}
        ).status_code
        == 422
    )


def test_batch_limit(tmp_path_factory, client, monkeypatch):
    from loyalty_spend.models.persistence import save_bundle

    path = save_bundle(
        client.app.state.bundle, tmp_path_factory.mktemp("lim") / "m.joblib"
    )
    monkeypatch.setenv("LOYALTY_MAX_BATCH", "2")
    small = TestClient(create_app(path))
    customers = [{**CUSTOMER, "customer_id": i} for i in range(3)]
    assert small.post("/predict", json={"customers": customers}).status_code == 413
