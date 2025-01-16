import pandas as pd
import pytest

from loyalty_spend.data.lineage import chance_agreement, compare_extracts, linkable


def test_chance_agreement_of_uniform_binary():
    a = pd.Series(["x", "y"] * 50)
    assert chance_agreement(a, a.sample(frac=1, random_state=0)) == pytest.approx(0.5)


def test_identical_extracts_are_linkable(train_df):
    table = compare_extracts(train_df, train_df.copy())
    assert (table["observed_agreement"] == 1).all()
    assert linkable(table)


def test_supplied_files_do_not_describe_the_same_customers(loyalty_df, train_df):
    table = compare_extracts(loyalty_df, train_df)
    assert table["n_shared_ids"].iloc[0] == len(train_df)
    assert table["kappa"].abs().max() < 0.1
    assert not linkable(table)
