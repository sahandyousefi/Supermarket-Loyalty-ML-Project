"""Encoding of tabular frames into tensors for neural networks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from loyalty_spend.data.schema import CATEGORICAL, NUMERIC, allowed_values


@dataclass
class TabularEncoder:
    """Standardize numerics and map categories to embedding indices.

    Index 0 is reserved for missing and unseen levels, so a category with
    ``k`` known levels needs an embedding table of size ``k + 1``. Missing
    numerics are replaced by the training median and flagged in an extra
    indicator column, mirroring the scikit-learn preprocessor.
    """

    numeric: list[str] = field(default_factory=lambda: list(NUMERIC))
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL))

    def fit(self, X: pd.DataFrame) -> TabularEncoder:
        num = X[self.numeric].astype(float)
        self.median_ = num.median().fillna(0.0)
        filled = num.fillna(self.median_)
        self.mean_ = filled.mean()
        self.std_ = filled.std(ddof=0).replace(0.0, 1.0)
        self.levels_ = {
            c: {lvl: i + 1 for i, lvl in enumerate(allowed_values(c))}
            for c in self.categorical
        }
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.levels_[c]) + 1 for c in self.categorical]

    @property
    def n_numeric(self) -> int:
        return 2 * len(self.numeric)

    def transform(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        num = X[self.numeric].astype(float)
        missing = num.isna().to_numpy(np.float32)
        z = ((num.fillna(self.median_) - self.mean_) / self.std_).to_numpy(np.float32)
        x_num = np.hstack([z, missing])
        x_cat = np.column_stack(
            [
                X[c].astype(object).map(self.levels_[c]).fillna(0).to_numpy(np.int64)
                for c in self.categorical
            ]
        )
        return torch.from_numpy(x_num), torch.from_numpy(x_cat)


def embedding_dim(cardinality: int, max_dim: int = 16) -> int:
    """Rule of thumb from fast.ai: grows sub-linearly with the cardinality."""
    return int(min(max_dim, round(1.6 * cardinality**0.56)))
