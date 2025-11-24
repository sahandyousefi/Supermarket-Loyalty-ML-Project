"""Reference values of the reported results and checks against them.

A reference file lists results by their path in ``summary.json`` (dots
for keys, ``[i]`` for list positions) with the value obtained when the
reference was made and a tolerance. Tolerances are relative and reflect
how a number is computed: closed-form statistics agree to rounding error
across platforms, while Monte Carlo results depend on the random stream
and on the order of floating point operations in the linear algebra
libraries.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_INDEX = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def lookup(summary: Mapping[str, Any], path: str) -> Any:
    """Value at ``path`` such as ``promotion.estimates[1].estimate``."""
    node: Any = summary
    for name, index in _INDEX.findall(path):
        node = node[int(index)] if index else node[name]
    return node


@dataclass(frozen=True)
class Reference:
    path: str
    value: float
    rel_tol: float
    abs_tol: float = 1e-12


def load_reference(path: str | Path) -> list[Reference]:
    entries = yaml.safe_load(Path(path).read_text())["results"]
    return [Reference(**entry) for entry in entries]


def write_reference(
    summary: Mapping[str, Any],
    paths: Mapping[str, float],
    out: str | Path,
    sections: Sequence[str],
) -> None:
    """Record the current values of ``paths`` with their relative tolerances."""
    results = [
        {"path": p, "value": float(lookup(summary, p)), "rel_tol": tol}
        for p, tol in paths.items()
    ]
    Path(out).write_text(
        yaml.safe_dump(
            {"sections": list(sections), "results": results}, sort_keys=False
        )
    )


def compare(
    summary: Mapping[str, Any], references: Sequence[Reference]
) -> pd.DataFrame:
    rows = []
    for ref in references:
        try:
            observed = float(lookup(summary, ref.path))
        except (KeyError, IndexError, TypeError):
            observed = float("nan")
        allowed = max(ref.abs_tol, ref.rel_tol * abs(ref.value))
        rows.append(
            {
                "path": ref.path,
                "reference": ref.value,
                "observed": observed,
                "difference": observed - ref.value,
                "allowed": allowed,
                "ok": bool(abs(observed - ref.value) <= allowed),
            }
        )
    return pd.DataFrame(rows)
