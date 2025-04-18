"""End-to-end analysis that produces the results tables and figures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from loyalty_spend.analysis.drivers import (
    effect_sizes,
    lmg_importance,
    spend_by_group,
    tukey_table,
    welch_anova,
)
from loyalty_spend.config import Config
from loyalty_spend.data.io import load_clean
from loyalty_spend.data.lineage import compare_extracts
from loyalty_spend.data.schema import FEATURES
from loyalty_spend.reporting import figures

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


def driver_section(cfg: Config, fig_dir: Path) -> tuple[str, dict]:
    clean, log = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    by_years = spend_by_group(clean, "loyalty_years", seed=cfg.cv.seed)
    anova = welch_anova(clean, "loyalty_years")
    sizes = effect_sizes(clean, "loyalty_years")
    tukey = tukey_table(clean, "loyalty_years")
    shares = lmg_importance(clean, FEATURES)
    figures.plot_group_means(by_years, "loyalty_years", fig_dir / "spend_by_tenure.png")
    figures.plot_driver_shares(shares, fig_dir / "driver_shares.png")
    text = "\n\n".join(
        [
            "## Data quality (loyalty.csv)",
            _md_table(log.to_frame()),
            "## Spend by loyalty tenure",
            _md_table(by_years, 2),
            f"Welch ANOVA: F = {anova['f']:.1f} on ({anova['df1']:.0f}, "
            f"{anova['df2']:.1f}) df, p = {anova['p_value']:.2e}. "
            f"Eta squared {sizes['eta_squared']:.3f}, "
            f"omega squared {sizes['omega_squared']:.3f}.",
            "### Tukey HSD",
            _md_table(tukey, 3),
            "### Shapley decomposition of R squared",
            _md_table(shares),
        ]
    )
    return text, {"welch": anova, **sizes, "lmg": shares.to_dict("records")}


def lineage_section(cfg: Config) -> str:
    loyalty, _ = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    table = compare_extracts(loyalty, train)
    return "## Lineage check: loyalty.csv against train.csv\n\n" + _md_table(table, 3)
