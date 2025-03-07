"""Training loop with early stopping for tabular networks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 400
    batch_size: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 40
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    seed: int = 0


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def gaussian_nll(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood of a Gaussian with predicted mean and log variance.

    The log variance is clamped so that a single badly fitted point cannot
    drive the variance to zero and the loss to minus infinity.
    """
    mean, log_var = output[:, 0], output[:, 1].clamp(-10.0, 10.0)
    return 0.5 * (log_var + (target - mean) ** 2 / log_var.exp()).mean()


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(output[:, 0], target)


def _lr_lambda(settings: TrainSettings):
    # Linear warmup, then cosine decay to 5% of the base rate.
    def f(epoch: int) -> float:
        if epoch < settings.warmup_epochs:
            return (epoch + 1) / settings.warmup_epochs
        progress = (epoch - settings.warmup_epochs) / max(
            1, settings.epochs - settings.warmup_epochs
        )
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f
