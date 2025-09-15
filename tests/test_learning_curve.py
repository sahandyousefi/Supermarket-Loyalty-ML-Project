from loyalty_spend.evaluation.cv import make_splits
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model
from loyalty_spend.robustness.learning_curve import has_plateaued, learning_curve


def test_error_falls_with_more_data_and_flattens(train_df):
    X, y = split_xy(train_df)
    curve = learning_curve(
        build_model("ols"),
        X,
        y,
        make_splits(len(y), 4, 1, 0),
        sizes=(0.05, 0.2, 0.6, 1.0),
    )
    assert curve["rmse"].iloc[0] > curve["rmse"].iloc[-1]
    assert (curve["lower"] <= curve["rmse"]).all() and (
        curve["rmse"] <= curve["upper"]
    ).all()
    assert has_plateaued(curve, tolerance=0.1)


def test_one_row_per_share_even_with_unequal_folds(train_df):
    X, y = split_xy(train_df)
    splits = make_splits(len(y) - 1, 3, 1, 0)
    curve = learning_curve(
        build_model("ols"), X.iloc[:-1], y.iloc[:-1], splits, sizes=(0.5, 1.0)
    )
    assert len(curve) == 2 and curve["std"].notna().all()
