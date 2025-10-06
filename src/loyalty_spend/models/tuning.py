"""Hyperparameter search with Optuna and nested cross-validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from loyalty_spend.evaluation.fold_cache import FoldCache
from loyalty_spend.evaluation.metrics import rmse
from loyalty_spend.models.registry import build_model

if TYPE_CHECKING:
    import optuna


def search_space(trial: optuna.trial.BaseTrial, name: str) -> dict[str, Any]:
    """Hyperparameters sampled for each model family."""
    if name == "hist_gbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 63, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 10.0, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 1200, step=100),
        }
    if name == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 63, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 100, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
        }
    if name == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 30.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
        }
    if name in ("mlp", "mlp_plr"):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "arch_params": {
                "width": trial.suggest_categorical("width", [64, 128, 256]),
                "depth": trial.suggest_int("depth", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            },
        }
    raise KeyError(f"no search space for '{name}'")


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 40,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[dict[str, Any], float]:
    """Minimize cross-validated RMSE with the TPE sampler.

    Returns the best parameters and their cross-validated RMSE.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    # Trials change only the model, so each fold is preprocessed once.
    cache = FoldCache(X, y, list(cv.split(X)))

    def objective(trial: optuna.Trial) -> float:
        model = build_model(name, **search_space(trial, name))
        scores = [
            rmse(y.iloc[te], cache.fit_predict(model, i)[1])
            for i, (_, te) in enumerate(cache.splits)
        ]
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    best = search_space(optuna.trial.FixedTrial(study.best_params), name)
    return best, float(study.best_value)


@dataclass
class NestedCVResult:
    outer_rmse: list[float] = field(default_factory=list)
    inner_rmse: list[float] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.outer_rmse))

    @property
    def optimism(self) -> float:
        """How much the tuning score understates the generalization error."""
        return float(np.mean(self.outer_rmse) - np.mean(self.inner_rmse))


def nested_cv(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 20,
    seed: int = 0,
) -> NestedCVResult:
    """Tune inside each outer training fold and score on the outer test fold.

    The outer score is an honest estimate of the tuned procedure; the inner
    score is what a single tuning run would report (Cawley and Talbot, 2010).
    """
    result = NestedCVResult()
    outer = KFold(outer_splits, shuffle=True, random_state=seed)
    for tr, te in outer.split(X):
        params, inner = tune(name, X.iloc[tr], y.iloc[tr], n_trials, inner_splits, seed)
        model = build_model(name, **params).fit(X.iloc[tr], y.iloc[tr])
        result.outer_rmse.append(rmse(y.iloc[te], model.predict(X.iloc[te])))
        result.inner_rmse.append(inner)
        result.params.append(params)
    return result
