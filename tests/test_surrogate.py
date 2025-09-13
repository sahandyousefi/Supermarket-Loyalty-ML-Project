from loyalty_spend.explain.surrogate import fit_surrogate
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model


def test_surrogate_of_the_champion(train_df):
    X, y = split_xy(train_df)
    model = build_model("ols").fit(X, y)
    shallow = fit_surrogate(model.predict, X, max_depth=2)
    deeper = fit_surrogate(model.predict, X, max_depth=4)
    assert 0 < shallow.fidelity <= deeper.fidelity <= 1
    assert "loyalty_years" in shallow.rules
    assert deeper.fidelity > 0.9
