"""Model card generated from a saved bundle.

The hand-written card in ``docs/model_card.md`` describes the champion of
the 1.0 analysis. A deployed model changes more often than that document,
so this module writes a short card for any bundle from what the bundle
itself records: model name, package and Python versions, training data
fingerprint, cross-validated metrics and the spread of its residuals,
following the headings of Mitchell et al. (2019).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from loyalty_spend.models.persistence import ModelBundle
from loyalty_spend.uncertainty.conformal import conformal_quantile


def _fingerprint(meta: dict[str, Any]) -> str:
    return str(meta.get("data_fingerprint") or meta.get("data") or "not recorded")


def render_model_card(bundle: ModelBundle, owner: str = "Sahand Yousefi") -> str:
    meta = bundle.metadata
    resid = np.asarray(bundle.residuals, float)
    lines = [
        f"# Model card: {bundle.name}",
        "",
        "## Model details",
        "",
        f"- **Owner:** {owner}",
        f"- **Model:** `{bundle.name}` from the model registry",
        f"- **Package version at training:** {bundle.package_version}",
        f"- **Python:** {meta.get('python', 'not recorded')}",
        f"- **Saved:** {meta.get('saved_at', 'not recorded')}",
        f"- **Training data fingerprint:** `{_fingerprint(meta)}`",
        f"- **Features:** {', '.join(bundle.features)}",
        "",
        "## Intended use",
        "",
        "Forecasting the annual spend of existing loyalty members to estimate reward"
        " cost. Not for decisions about individual members.",
        "",
        "## Metrics",
        "",
    ]
    cv = meta.get("cv") or {}
    if cv:
        lines += ["| Metric | Value |", "| --- | --- |"]
        for key, value in cv.items():
            if isinstance(value, int | float):
                lines.append(f"| {key} | {value:.4f} |")
        lines.append("")
    else:
        lines += ["No cross-validated metrics were recorded with this bundle.", ""]
    if resid.size:
        half90 = conformal_quantile(np.abs(resid), 0.1)
        lines += [
            "## Uncertainty",
            "",
            f"- Out-of-fold residuals: {resid.size}, mean {resid.mean():.4f},"
            f" standard deviation {resid.std(ddof=1):.4f}",
            f"- Half-width of the 90% conformal interval: {half90:.4f}",
            "",
        ]
    lines += [
        "## Caveats",
        "",
        "- Intervals assume new members resemble the training members; run"
        " `loyalty-spend monitor` on every batch before scoring it.",
        "- See `docs/model_card.md` and `docs/methodology.md` for the analysis"
        " behind the model choice.",
    ]
    return "\n".join(lines) + "\n"


def write_model_card(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_card(bundle))
    return path
