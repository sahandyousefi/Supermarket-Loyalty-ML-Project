from pathlib import Path

import pytest

from loyalty_spend.data.io import load_clean
from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import Scenario, simulate

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


@pytest.fixture(scope="session")
def sim_df(train_df):
    return simulate(train_df, Scenario("test", noise=0.1), n=600, seed=1).frame


@pytest.fixture(scope="session")
def xy(sim_df):
    return sim_df[FEATURES], sim_df["spend"]
