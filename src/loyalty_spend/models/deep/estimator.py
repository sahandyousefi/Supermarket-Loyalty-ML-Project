"""scikit-learn compatible wrappers around the tabular networks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from loyalty_spend.models.deep.networks import ARCHITECTURES
from loyalty_spend.models.deep.tabular import TabularEncoder
from loyalty_spend.models.deep.training import (
    TrainSettings,
    fit_network,
    gaussian_nll,
    mse_loss,
)
from loyalty_spend.models.registry import register


class TorchTabularRegressor(RegressorMixin, BaseEstimator):
    """Neural regressor with an internal validation split for early stopping.

    The target is standardized for training and transformed back for
    prediction. With ``probabilistic=True`` the network predicts a mean and a
    log variance and is trained with the Gaussian negative log likelihood.

    Parameters
    ----------
    architecture : {"mlp", "ft_transformer"}
    arch_params : dict, optional
        Keyword arguments of the network class.
    epochs, batch_size, learning_rate, weight_decay, patience
        Optimizer and early stopping settings.
    val_fraction : float
        Share of the training rows held out for early stopping.
    probabilistic : bool
        Gaussian output head.
    seed : int
        Seed of the split, initialization and batching.
    """

    def __init__(
        self,
        architecture: str = "mlp",
        arch_params: dict[str, Any] | None = None,
        epochs: int = 400,
        batch_size: int = 64,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        patience: int = 40,
        val_fraction: float = 0.15,
        probabilistic: bool = False,
        seed: int = 0,
    ) -> None:
        self.architecture = architecture
        self.arch_params = arch_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.probabilistic = probabilistic
        self.seed = seed

    def _settings(self) -> TrainSettings:
        return TrainSettings(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            seed=self.seed,
        )

    def _tensors(self, X: pd.DataFrame, y: np.ndarray | None = None):
        x_num, x_cat = self.encoder_.transform(X)
        if y is None:
            return x_num, x_cat
        target = torch.as_tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        return x_num, x_cat, target

    def fit(self, X: pd.DataFrame, y) -> TorchTabularRegressor:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}")
        torch.manual_seed(self.seed)
        y = np.asarray(y, float)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, random_state=self.seed
        )
        self.encoder_ = TabularEncoder().fit(X_tr)
        self.y_mean_, self.y_std_ = float(y_tr.mean()), float(y_tr.std() or 1.0)
        self.network_ = ARCHITECTURES[self.architecture](
            n_numeric=self.encoder_.n_numeric,
            cardinalities=self.encoder_.cardinalities,
            n_outputs=2 if self.probabilistic else 1,
            **(self.arch_params or {}),
        )
        self.history_ = fit_network(
            self.network_,
            self._tensors(X_tr, y_tr),
            self._tensors(X_val, y_val),
            self._settings(),
            loss_fn=gaussian_nll if self.probabilistic else mse_loss,
        )
        return self

    @torch.no_grad()
    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        self.network_.eval()
        return self.network_(*self._tensors(X)).numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._raw(X)[:, 0] * self.y_std_ + self.y_mean_

    def predict_dist(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and variance on the original scale."""
        if not self.probabilistic:
            raise AttributeError("predict_dist needs probabilistic=True")
        out = self._raw(X)
        mean = out[:, 0] * self.y_std_ + self.y_mean_
        var = np.exp(np.clip(out[:, 1], -10, 10)) * self.y_std_**2
        return mean, var


class DeepEnsembleRegressor(RegressorMixin, BaseEstimator):
    """Average of independently seeded Gaussian networks.

    The predictive distribution is the uniform mixture of the members, with
    mean ``mu = mean_m mu_m`` and variance
    ``mean_m (sigma_m^2 + mu_m^2) - mu^2`` (Lakshminarayanan et al., 2017).
    The first term is aleatoric noise, the spread of the member means is
    epistemic uncertainty.
    """

    def __init__(
        self,
        n_members: int = 5,
        architecture: str = "mlp",
        arch_params: dict[str, Any] | None = None,
        epochs: int = 400,
        patience: int = 40,
        seed: int = 0,
    ) -> None:
        self.n_members = n_members
        self.architecture = architecture
        self.arch_params = arch_params
        self.epochs = epochs
        self.patience = patience
        self.seed = seed

    def fit(self, X: pd.DataFrame, y) -> DeepEnsembleRegressor:
        self.members_ = [
            TorchTabularRegressor(
                architecture=self.architecture,
                arch_params=self.arch_params,
                epochs=self.epochs,
                patience=self.patience,
                probabilistic=True,
                seed=self.seed + m,
            ).fit(X, y)
            for m in range(self.n_members)
        ]
        return self

    def predict_components(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Aleatoric and epistemic variance per row."""
        means, variances = zip(*(m.predict_dist(X) for m in self.members_))
        means, variances = np.vstack(means), np.vstack(variances)
        return variances.mean(axis=0), means.var(axis=0)

    def predict_dist(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        means = np.vstack([m.predict_dist(X)[0] for m in self.members_])
        aleatoric, epistemic = self.predict_components(X)
        return means.mean(axis=0), aleatoric + epistemic

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_dist(X)[0]


@register("mlp", family="deep", requires="torch")
def mlp(**params: Any) -> TorchTabularRegressor:
    return TorchTabularRegressor(architecture="mlp", **params)


@register("ft_transformer", family="deep", requires="torch")
def ft_transformer(**params: Any) -> TorchTabularRegressor:
    defaults = dict(learning_rate=1e-3, weight_decay=1e-5, batch_size=128)
    return TorchTabularRegressor(
        architecture="ft_transformer", **{**defaults, **params}
    )


@register("deep_ensemble", family="deep", requires="torch")
def deep_ensemble(**params: Any) -> DeepEnsembleRegressor:
    return DeepEnsembleRegressor(**params)


@register("mlp_plr", family="deep", requires="torch")
def mlp_plr(**params: Any) -> TorchTabularRegressor:
    """MLP with periodic embeddings of the numeric features."""
    arch = {"numeric_embedding": "plr", **params.pop("arch_params", {})}
    return TorchTabularRegressor(architecture="mlp", arch_params=arch, **params)
