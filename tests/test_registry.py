import numpy as np
import pytest

from loyalty_spend.models.persistence import ModelBundle, data_fingerprint
from loyalty_spend.models.registry import build_model
from loyalty_spend.ops.registry import ModelRegistry


@pytest.fixture
def bundle(xy):
    X, y = xy
    model = build_model("ols").fit(X, y)
    return ModelBundle(
        "ols",
        model,
        y.to_numpy() - model.predict(X),
        metadata={"data_fingerprint": data_fingerprint(X)},
    )


def test_versions_increase_and_round_trip(bundle, xy, tmp_path):
    registry = ModelRegistry(tmp_path / "reg")
    first = registry.register(bundle, {"rmse": 1.5})
    second = registry.register(bundle, {"rmse": 1.4}, notes="retrained")
    assert (first.version, second.version) == (1, 2)
    assert second.tag == "v0002"
    assert [r.version for r in registry.records()] == [1, 2]
    loaded = registry.load(2)
    X, _ = xy
    np.testing.assert_allclose(loaded.predict(X), bundle.predict(X))
    assert registry.get(1).metrics == {"rmse": 1.5}
    assert registry.get(2).data_fingerprint == bundle.metadata["data_fingerprint"]


def test_missing_version(tmp_path):
    with pytest.raises(KeyError):
        ModelRegistry(tmp_path).load(3)
