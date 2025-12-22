"""Wall-clock time of the main computations on the supplied data.

Each task runs ``--repeats`` times after one warm-up run and the median is
reported, so import and first-call costs are excluded.

Usage::

    python scripts/benchmark_runtime.py --repeats 3
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from loyalty_spend.bayes.hierarchical import fit_hierarchical
from loyalty_spend.business.design import search_thresholds
from loyalty_spend.business.rewards import (
    RewardSchedule,
    forecast_liability,
    residual_sampler,
)
from loyalty_spend.config import load_config
from loyalty_spend.data.io import load_clean
from loyalty_spend.evaluation.cv import cross_validate, make_splits
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model
from loyalty_spend.synthetic.conditional import ConditionalCopulaSynthesizer


def tasks() -> dict[str, Callable[[], Any]]:
    cfg = load_config()
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    test, _ = load_clean(
        cfg.paths.raw(cfg.paths.test_file), impute=False, require_target=False
    )
    X, y = split_xy(train)
    splits = make_splits(len(y), 5, 1, 0)
    ols = build_model("ols").fit(X, y)
    pred = ols.predict(test)
    resid = y.to_numpy() - cross_validate("ols", build_model("ols"), X, y, splits).oof
    schedule = RewardSchedule(cfg.rewards.tiers)

    def cv(name: str, **params: Any) -> Callable[[], Any]:
        return lambda: cross_validate(name, build_model(name, **params), X, y, splits)

    return {
        "5-fold CV, OLS": cv("ols"),
        "5-fold CV, weighted by month": cv("wls_month"),
        "5-fold CV, histogram GBM": cv("hist_gbm"),
        "5-fold CV, Gaussian process, one start": cv("gp", n_restarts_optimizer=0),
        "reward cost, 5,000 draws x 250 members": lambda: forecast_liability(
            np.arange(len(pred)), pred, resid, schedule, 5000, 0
        ),
        "threshold search, 2,000 draws": lambda: search_thresholds(
            residual_sampler(pred, resid), schedule, budget=1300.0, n_simulations=2000
        ),
        "Gibbs sampler, 4 chains x 1,500 sweeps": lambda: fit_hierarchical(X, y),
        "conditional copula, fit and 1,000 rows": lambda: (
            ConditionalCopulaSynthesizer("loyalty_years")
            .fit(train.drop(columns="customer_id"))
            .sample(1000)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows = []
    for name, task in tasks().items():
        task()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            task()
            times.append(time.perf_counter() - start)
        rows.append({"task": name, "median_seconds": float(np.median(times))})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
