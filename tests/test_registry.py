import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from loyalty_spend.models.persistence import ModelBundle, data_fingerprint
from loyalty_spend.models.registry import build_model
from loyalty_spend.ops.registry import ModelRegistry, promote_if_better


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


def test_promotion_keeps_one_production_version(bundle, tmp_path):
    registry = ModelRegistry(tmp_path / "reg")
    for _ in range(3):
        registry.register(bundle)
    registry.promote(1)
    registry.promote(2)
    assert registry.current().version == 2
    assert registry.get(1).stage == "archived"
    registry.promote(3, "staging")
    assert registry.current("staging").version == 3
    with pytest.raises(ValueError):
        registry.promote(3, "live")


def test_rollback_restores_the_previous_production(bundle, tmp_path):
    registry = ModelRegistry(tmp_path / "reg")
    registry.register(bundle)
    registry.register(bundle)
    with pytest.raises(RuntimeError):
        registry.rollback()
    registry.promote(1)
    registry.promote(2)
    restored = registry.rollback()
    assert restored.version == 1 and registry.current().version == 1
    assert registry.get(2).stage == "archived"


def test_challenger_is_promoted_only_when_it_wins(xy, tmp_path):
    X, y = xy
    good = build_model("ols").fit(X, y)
    registry = ModelRegistry(tmp_path / "reg")
    weak = ModelBundle("mean", DummyRegressor().fit(X, y), np.zeros(3))
    v1 = registry.register(weak).version
    promoted, _ = promote_if_better(registry, v1, X, y)
    assert promoted and registry.current().version == v1
    v2 = registry.register(ModelBundle("ols", good, np.zeros(3))).version
    promoted, summary = promote_if_better(registry, v2, X, y)
    assert promoted and registry.current().version == v2
    assert summary["challenger_rmse"] < summary["champion_rmse"]
    v3 = registry.register(ModelBundle("ols", good, np.zeros(3))).version
    promoted, _ = promote_if_better(registry, v3, X, y, margin=0.01)
    assert not promoted and registry.current().version == v2
    assert registry.get(v3).stage == "archived"
    assert "challenge_p_value" in registry.get(v3).metrics
