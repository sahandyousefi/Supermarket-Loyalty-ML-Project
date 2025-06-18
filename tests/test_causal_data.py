import numpy as np
import pandas as pd
import pytest

from loyalty_spend.causal.data import PRE_TREATMENT, CausalData, from_frame


def test_study_from_the_training_file(train_df):
    study = from_frame(train_df)
    assert list(study.X.columns) == PRE_TREATMENT
    assert study.n_treated + study.n_control == len(study.y)
    assert 0 < study.n_treated < len(study.y)
    expected = (
        train_df.loc[train_df["promotion"] == "Yes", "spend"].mean()
        - train_df.loc[train_df["promotion"] == "No", "spend"].mean()
    )
    assert study.naive_difference() == pytest.approx(expected)


def test_invalid_studies_are_rejected():
    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        CausalData(X, np.array([0, 1]), np.zeros(3))
    with pytest.raises(ValueError):
        CausalData(X, np.array([0, 2, 1]), np.zeros(3))
    with pytest.raises(ValueError):
        CausalData(X, np.ones(3, dtype=int), np.zeros(3))
    with pytest.raises(ValueError):
        from_frame(pd.DataFrame({"promotion": ["Yes"], "spend": [1.0]}), ["promotion"])
