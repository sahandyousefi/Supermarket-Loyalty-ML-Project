"""Utility of synthetic data: train on synthetic, test on real.

For every fold of the real data, a synthesiser is fitted on the training
part only, a synthetic sample of the same size is drawn, and a model is
fitted once on the real and once on the synthetic training data. Both are
scored on the real held-out part (Esteban, Hyland and Ratsch, 2017). The
ratio of the two RMSEs measures how much predictive information the
synthetic data keep; a synthesiser that never sees the held-out members
cannot inflate it by copying them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from loyalty_spend.arrays import FloatArray, Split
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model


def _rmse(a: FloatArray, b: FloatArray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def train_synthetic_test_real(
    data: pd.DataFrame,
    synthesizers: dict[str, Callable[[int], Any]],
    splits: list[Split],
    models: tuple[str, ...] = ("ols",),
) -> pd.DataFrame:
    """Fold-level RMSE on real held-out data for each synthesiser and model.

    ``synthesizers`` maps a name to a factory that takes a seed and returns
    an object with ``fit(frame)`` and ``sample(n)``.
    """
    rows = []
    for k, (tr, te) in enumerate(splits):
        train, test = data.iloc[tr], data.iloc[te]
        X_te, y_te = split_xy(test)
        sources = {"real": train}
        for name, factory in synthesizers.items():
            sources[name] = factory(k).fit(train).sample(len(train))
        for model in models:
            for source, frame in sources.items():
                X, y = split_xy(frame.dropna(subset=["spend"]))
                pred = build_model(model).fit(X, y).predict(X_te)
                rows.append(
                    {
                        "fold": k,
                        "model": model,
                        "source": source,
                        "rmse": _rmse(pred, y_te),
                    }
                )
    table = pd.DataFrame(rows)
    real = table[table["source"] == "real"].set_index(["fold", "model"])["rmse"]
    table["ratio_to_real"] = (
        table["rmse"].to_numpy()
        / real.loc[list(zip(table["fold"], table["model"], strict=True))].to_numpy()
    )
    return table


def utility_summary(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["model", "source"])
        .agg(
            rmse=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            ratio_to_real=("ratio_to_real", "mean"),
        )
        .reset_index()
    )
