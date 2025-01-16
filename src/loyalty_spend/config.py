"""Typed configuration loaded from YAML with dataclass defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
