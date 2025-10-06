"""Repeated cross-validation and paired model comparison."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedKFold

from loyalty_spend.evaluation.fold_cache import FoldCache
from loyalty_spend.evaluation.metrics import regression_report


@dataclass
class CVResult:
    """Fold metrics and out-of-fold predictions of one model."""

    name: str
    folds: pd.DataFrame
    oof: np.ndarray
    n_train: float
    n_test: float

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"model": self.name}  # type: ignore[dict-item]
        for col in ("rmse", "mae", "mape", "r2"):
            out[f"{col}_mean"] = float(self.folds[col].mean())
            out[f"{col}_std"] = float(self.folds[col].std(ddof=1))
        return out


def make_splits(
    n: int, n_splits: int = 5, n_repeats: int = 3, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed splits so that every model is evaluated on identical folds."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(n)))


def _fit_predict(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    tr: np.ndarray,
    te: np.ndarray,
) -> tuple[BaseEstimator, np.ndarray]:
    model = clone(estimator)
    model.fit(X.iloc[tr], y.iloc[tr])
    return model, np.asarray(model.predict(X.iloc[te]), float)


def cross_validate(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fit_hook: Callable[[BaseEstimator], None] | None = None,
    n_jobs: int = 1,
    cache: FoldCache | None = None,
) -> CVResult:
    """Fit a fresh clone on each split and score it on the held-out fold.

    Out-of-fold predictions are averaged over repeats, so ``oof`` holds one
    prediction per row. With ``n_jobs`` other than 1 the folds are fitted
    in separate processes; every estimator carries its own seed, so the
    results do not depend on the number of jobs. ``fit_hook`` always runs
    in the calling process. A ``cache`` built on the same data and splits
    reuses fitted preprocessing across models and calls.
    """
    if cache is not None:
        if cache.splits is not splits:
            raise ValueError("the cache was built for other splits")
        fitted = [cache.fit_predict(estimator, i) for i in range(len(splits))]
    elif n_jobs == 1:
        fitted = [_fit_predict(estimator, X, y, tr, te) for tr, te in splits]
    else:
        fitted = Parallel(n_jobs=n_jobs)(
            delayed(_fit_predict)(estimator, X, y, tr, te) for tr, te in splits
        )
    rows = []
    oof_sum = np.zeros(len(y))
    oof_n = np.zeros(len(y))
    for i, ((_, te), (model, pred)) in enumerate(zip(splits, fitted, strict=True)):
        if fit_hook is not None:
            fit_hook(model)
        oof_sum[te] += pred
        oof_n[te] += 1
        rows.append({"fold": i, **regression_report(y.iloc[te], pred)})
    oof = oof_sum / np.maximum(oof_n, 1)
    n_test = float(np.mean([len(te) for _, te in splits]))
    return CVResult(name, pd.DataFrame(rows), oof, len(y) - n_test, n_test)


def corrected_ttest(
    diffs: np.ndarray, n_train: float, n_test: float
) -> tuple[float, float]:
    """Corrected resampled t-test of Nadeau and Bengio (2003).

    Fold scores share most of their training data, so the naive paired
    t-test underestimates the variance. The correction replaces ``1/J`` by
    ``1/J + n_test/n_train``. Returns the statistic and two-sided p-value.
    """
    d = np.asarray(diffs, float)
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt((1 / j + n_test / n_train) * var)
    return float(stat), float(2 * student_t.sf(abs(stat), df=j - 1))


def compare_results(
    results: list[CVResult], reference: str, metric: str = "rmse"
) -> pd.DataFrame:
    """Summary table with the paired difference to a reference model."""
    by_name = {r.name: r for r in results}
    if reference not in by_name:
        raise KeyError(f"reference model '{reference}' not in results")
    base = by_name[reference]
    ref = base.folds[metric].to_numpy()
    rows = []
    for r in results:
        d = r.folds[metric].to_numpy() - ref
        stat, p = (
            (np.nan, np.nan)
            if r.name == reference
            else corrected_ttest(d, base.n_train, base.n_test)
        )
        rows.append(
            {
                **r.summary(),
                f"{metric}_diff_vs_{reference}": float(d.mean()),
                "t_stat": stat,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows).sort_values(f"{metric}_mean").reset_index(drop=True)
