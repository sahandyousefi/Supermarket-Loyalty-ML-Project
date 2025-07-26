"""One monitoring check of a new scoring batch against the training data.

The check combines the data contract, per-feature drift, the domain
classifier and prediction drift into a single status: ``ok`` when nothing
moved, ``warn`` for a moderate shift, and ``alert`` for a contract error, a
large shift or a domain classifier above its alert level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from loyalty_spend.config import MonitoringConfig
from loyalty_spend.data.contracts import TableContract, check_contract
from loyalty_spend.models.persistence import ModelBundle
from loyalty_spend.monitoring.domain import domain_classifier_test
from loyalty_spend.monitoring.drift import (
    DriftThresholds,
    distribution_shift,
    drift_report,
)


@dataclass
class MonitoringResult:
    status: str
    reasons: list[str]
    drift: pd.DataFrame
    domain_auc: float
    prediction_shift: dict[str, float]
    contract_violations: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "domain_auc": self.domain_auc,
            "prediction_shift": self.prediction_shift,
            "drift": self.drift.to_dict("records"),
            "contract_violations": self.contract_violations.to_dict("records"),
        }


def monitor_batch(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    bundle: ModelBundle,
    cfg: MonitoringConfig | None = None,
    contract: TableContract | None = None,
    seed: int = 0,
) -> MonitoringResult:
    cfg = cfg or MonitoringConfig()
    reasons: list[str] = []
    level = 0
    violations = pd.DataFrame()
    if contract is not None:
        report = check_contract(batch, contract)
        violations = report.to_frame()
        if not report.ok:
            reasons.append(f"{len(report.errors)} contract errors")
            level = 2
    drift = drift_report(
        reference, batch, n_bins=cfg.n_bins, thresholds=DriftThresholds.from_config(cfg)
    )
    for status, lvl in (("large", 2), ("moderate", 1)):
        moved = drift.loc[drift["status"] == status, "feature"].tolist()
        if moved:
            reasons.append(f"{status} drift in {', '.join(moved)}")
            level = max(level, lvl)
    auc = domain_classifier_test(reference, batch, seed=seed).auc
    if auc >= cfg.domain_auc_alert:
        reasons.append(f"domain classifier AUC {auc:.3f}")
        level = 2
    shift = distribution_shift(bundle.predict(reference), bundle.predict(batch))
    if shift["psi"] >= cfg.psi_large and shift["p_value"] < cfg.fdr:
        reasons.append(f"prediction PSI {shift['psi']:.3f}")
        level = max(level, 2)
    status = ("ok", "warn", "alert")[level]
    return MonitoringResult(status, reasons, drift, auc, shift, violations)
