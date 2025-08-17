import json

import numpy as np
import pytest

from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.llm.client import MockLLM
from loyalty_spend.llm.reasons import (
    explain_forecast,
    forecast_reasons,
    template_explanation,
)
from loyalty_spend.models.registry import build_model


@pytest.fixture(scope="module")
def reasons(train_df):
    X, y = split_xy(train_df)
    model = build_model("ols").fit(X, y)
    return forecast_reasons(model, X, X.iloc[[3]])


def test_contributions_add_up_to_the_forecast(reasons):
    total = reasons.baseline + reasons.contributions.sum()
    assert total == pytest.approx(reasons.prediction, abs=1e-6)
    assert reasons.top(2).abs().is_monotonic_decreasing


def test_template_and_model_explanations(reasons):
    text = template_explanation(reasons)
    assert f"{reasons.prediction:.2f}" in text
    top_name, top_value = next(iter(reasons.top(1).items()))
    good = json.dumps(
        {
            "explanation": f"The member's forecast is {reasons.prediction:.2f}; "
            f"the biggest factor moves it by {abs(top_value):.2f}."
        }
    )
    bad = json.dumps({"explanation": "Tenure adds 12.34."})
    replies = iter([bad, good])
    llm = MockLLM(lambda s, u: next(replies))
    out = explain_forecast(llm, reasons)
    assert f"{reasons.prediction:.2f}" in out
    assert "12.34" in llm.calls[1][1]
    assert np.isfinite(reasons.baseline)
