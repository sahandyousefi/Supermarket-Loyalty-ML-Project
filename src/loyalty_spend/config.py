"""Typed configuration loaded from YAML with dataclass defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: str = "data/raw"
    loyalty_file: str = "loyalty.csv"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    output_dir: str = "artifacts"
    report_dir: str = "reports"

    def raw(self, name: str) -> Path:
        return Path(self.raw_dir) / name


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_splits < 2 or self.n_repeats < 1:
            raise ValueError("cv.n_splits must be >= 2 and cv.n_repeats >= 1")


@dataclass(frozen=True)
class DeepConfig:
    epochs: int = 400
    batch_size: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 40
    val_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("deep.val_fraction must be in (0, 0.5)")


@dataclass(frozen=True)
class RewardConfig:
    # (minimum annual spend, reward rate) pairs; the rate applies to the
    # whole annual spend once the threshold is reached.
    tiers: tuple[tuple[float, float], ...] = (
        (0.0, 0.00),
        (100.0, 0.02),
        (125.0, 0.03),
        (140.0, 0.05),
    )
    n_simulations: int = 5000
    seed: int = 7


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.1
    calibration_size: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("conformal.alpha must be in (0, 1)")


@dataclass(frozen=True)
class MonitoringConfig:
    # PSI bands follow the common credit-scoring convention; a shift is only
    # flagged when the drift test is also significant at the given FDR.
    psi_moderate: float = 0.1
    psi_large: float = 0.25
    fdr: float = 0.05
    n_bins: int = 10
    domain_auc_alert: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 < self.psi_moderate < self.psi_large:
            raise ValueError("monitoring: need 0 < psi_moderate < psi_large")
        if not 0.0 < self.fdr < 1.0:
            raise ValueError("monitoring.fdr must be in (0, 1)")
        if self.n_bins < 2:
            raise ValueError("monitoring.n_bins must be at least 2")
        if not 0.5 < self.domain_auc_alert < 1.0:
            raise ValueError("monitoring.domain_auc_alert must be in (0.5, 1)")


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


_SECTIONS = {
    "paths": PathsConfig,
    "cv": CVConfig,
    "deep": DeepConfig,
    "rewards": RewardConfig,
    "conformal": ConformalConfig,
    "monitoring": MonitoringConfig,
}


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if cls is RewardConfig and "tiers" in values:
        values["tiers"] = tuple(tuple(float(v) for v in t) for t in values["tiers"])
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Read a YAML file; missing sections and keys fall back to defaults."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text()) if path.exists() else None
    raw = raw or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    return Config(**{k: _build(cls, raw.get(k)) for k, cls in _SECTIONS.items()})
