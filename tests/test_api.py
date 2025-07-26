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
