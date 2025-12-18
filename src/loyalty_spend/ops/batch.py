"""Scoring a batch of customers with intervals and input checks.

The bundle's out-of-fold residuals are exchangeable with the errors on new
customers drawn from the same population, so their conformal quantile gives
a prediction interval for every score at no extra cost. The batch is
checked against the scoring contract first; with ``strict`` a violation
stops the run, otherwise the violations are returned with the scores. A
bundle whose model gives an error SD per member (the month-weighted model)
carries out-of-fold residuals scaled by that SD, and its intervals are
normalized conformal intervals whose width follows the member's month.
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
    scaled = bundle.metadata.get("scaled_residuals")
    if scaled is not None and hasattr(bundle.model, "predict_sd"):
        # Normalized scores: the quantile is scaled by each member's error SD.
        from loyalty_spend.features.preprocess import feature_frame

        q = conformal_quantile(np.abs(np.asarray(scaled, float)), alpha)
        half = q * bundle.model.predict_sd(feature_frame(frame))
    else:
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
