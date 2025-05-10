"""Training loop with early stopping for tabular networks."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

Batch = tuple[torch.Tensor, ...]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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


def _lr_lambda(settings: TrainSettings) -> Callable[[int], float]:
    # Linear warmup, then cosine decay to 5% of the base rate.
    def f(epoch: int) -> float:
        if epoch < settings.warmup_epochs:
            return (epoch + 1) / settings.warmup_epochs
        progress = (epoch - settings.warmup_epochs) / max(
            1, settings.epochs - settings.warmup_epochs
        )
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


@torch.no_grad()
def evaluate(model: nn.Module, tensors: Batch, loss_fn: LossFn) -> float:
    model.eval()
    x_num, x_cat, y = tensors
    return float(loss_fn(model(x_num, x_cat), y))


def fit_network(
    model: nn.Module,
    train: Batch,
    val: Batch,
    settings: TrainSettings,
    loss_fn: LossFn = mse_loss,
) -> History:
    """Train with AdamW and keep the weights of the best validation epoch."""
    set_seed(settings.seed)
    loader = DataLoader(
        TensorDataset(*train),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(settings))
    history = History()
    best_state, best_loss, stale = copy.deepcopy(model.state_dict()), math.inf, 0

    for epoch in range(settings.epochs):
        model.train()
        total, count = 0.0, 0
        for x_num, x_cat, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_num, x_cat), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        scheduler.step()
        history.train_loss.append(total / count)
        history.val_loss.append(evaluate(model, val, loss_fn))

        if history.val_loss[-1] < best_loss - 1e-6:
            best_loss, stale = history.val_loss[-1], 0
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
        else:
            stale += 1
            if stale >= settings.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return history
