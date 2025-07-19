"""Scoring a batch of customers with intervals and input checks.

The bundle's out-of-fold residuals are exchangeable with the errors on new
customers drawn from the same population, so their conformal quantile gives
a prediction interval for every score at no extra cost. The batch is
checked against the scoring contract first; with ``strict`` a violation
stops the run, otherwise the violations are returned with the scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from loyalty_spend.data.contracts import ContractReport, TableContract, check_contract
from loyalty_spend.data.schema import ID
from loyalty_spend.models.persistence import ModelBundle
from loyalty_spend.uncertainty.conformal import conformal_quantile


@dataclass
class BatchResult:
    scores: pd.DataFrame
    contract: ContractReport | None


def score_batch(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    alpha: float = 0.1,
    contract: TableContract | None = None,
    strict: bool = True,
) -> BatchResult:
    report = check_contract(frame, contract) if contract is not None else None
    if report is not None and strict:
        report.raise_for_errors()
    pred = bundle.predict(frame)
    half = conformal_quantile(np.abs(np.asarray(bundle.residuals, float)), alpha)
    scores = pd.DataFrame(
        {
            ID: frame[ID].to_numpy(),
            "prediction": pred,
            "lower": np.clip(pred - half, 0.0, None),
            "upper": pred + half,
        }
    )
    return BatchResult(scores, report)
