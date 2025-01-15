from pathlib import Path

import pytest

from loyalty_spend.data.io import load_clean

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW


@pytest.fixture(scope="session")
def train_df():
    df, _ = load_clean(RAW / "train.csv", impute=False)
    return df


@pytest.fixture(scope="session")
def loyalty_df():
    df, _ = load_clean(RAW / "loyalty.csv", impute=False)
    return df
