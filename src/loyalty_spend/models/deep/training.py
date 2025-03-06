"""Training loop with early stopping for tabular networks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


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
