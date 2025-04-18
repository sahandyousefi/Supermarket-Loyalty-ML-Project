"""End-to-end analysis that produces the results tables and figures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MODELS = (
    "ols",
    "ridge",
    "elastic_net",
    "huber",
    "gamma_glm",
    "spline_ridge",
    "random_forest",
    "hist_gbm",
    "xgboost",
    "lightgbm",
    "mlp",
    "mlp_plr",
    "ft_transformer",
)
STUDY_MODELS = ("ols", "spline_ridge", "hist_gbm", "lightgbm", "mlp", "mlp_plr")


@dataclass(frozen=True)
class ReportSettings:
    models: Sequence[str] = DEFAULT_MODELS
    study_models: Sequence[str] = STUDY_MODELS
    study_n: int = 2000
    study_splits: int = 5
    conformal_scenario: str = "noisy"
    conformal_test_n: int = 5000


def _md_table(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt(v) -> str:
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(False)]
    return "\n".join([head, sep, *rows])
