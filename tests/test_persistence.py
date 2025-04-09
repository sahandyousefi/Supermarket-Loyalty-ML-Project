import numpy as np
import pytest

from loyalty_spend.models.persistence import (
    ModelBundle,
    data_fingerprint,
    load_bundle,
    save_bundle,
)
from loyalty_spend.models.registry import build_model


def test_round_trip(tmp_path, xy):
    X, y = xy
    model = build_model("ridge").fit(X, y)
    bundle = ModelBundle("ridge", model, residuals=np.zeros(3))
    loaded = load_bundle(save_bundle(bundle, tmp_path / "m.joblib"))
    assert np.allclose(loaded.predict(X), model.predict(X))
    assert "saved_at" in loaded.metadata


def test_load_rejects_other_objects(tmp_path):
    import joblib

    joblib.dump({"a": 1}, tmp_path / "x.joblib")
    with pytest.raises(TypeError):
        load_bundle(tmp_path / "x.joblib")


def test_fingerprint_changes_with_data(train_df):
    a = data_fingerprint(train_df)
    b = data_fingerprint(train_df.assign(spend=train_df["spend"] + 1))
    assert a != b and len(a) == 12
