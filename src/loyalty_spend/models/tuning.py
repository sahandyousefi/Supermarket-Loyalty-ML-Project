"""Hyperparameter search with Optuna and nested cross-validation."""

from __future__ import annotations

from typing import Any


def search_space(trial, name: str) -> dict[str, Any]:
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
