import numpy as np

from loyalty_spend.models.persistence import ModelBundle
from loyalty_spend.models.registry import build_model
from loyalty_spend.ops.model_card import render_model_card, write_model_card


def test_card_reports_metadata_and_uncertainty(xy, tmp_path):
    X, y = xy
    bundle = ModelBundle(
        "ols",
        build_model("ols").fit(X, y),
        np.linspace(-1, 1, 101),
        metadata={"cv": {"rmse_mean": 0.31, "model": "ols"}, "data": "abc123def456"},
    )
    text = render_model_card(bundle)
    assert text.startswith("# Model card: ols")
    assert "`abc123def456`" in text
    assert "| rmse_mean | 0.3100 |" in text
    assert "90% conformal interval" in text
    path = write_model_card(bundle, tmp_path / "card" / "MODEL_CARD.md")
    assert path.read_text() == text


def test_card_without_metrics(xy):
    X, y = xy
    bundle = ModelBundle("ols", build_model("ols").fit(X, y), np.array([]))
    text = render_model_card(bundle)
    assert "No cross-validated metrics" in text and "## Uncertainty" not in text
