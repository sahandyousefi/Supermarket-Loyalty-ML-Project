"""Experiment tracking with MLflow when it is installed.

Without MLflow the same calls append one JSON line per run to a local file,
so every command can log its parameters and metrics without a hard
dependency, and the file can be imported into a tracking server later.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def mlflow_available() -> bool:
    return importlib.util.find_spec("mlflow") is not None


def log_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str | Path] | None = None,
    experiment: str = "loyalty-spend",
    local_path: str | Path = "artifacts/runs.jsonl",
    use_mlflow: bool | None = None,
) -> str:
    """Record one run and return where it went ("mlflow" or the file path)."""
    use_mlflow = mlflow_available() if use_mlflow is None else use_mlflow
    if use_mlflow:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            for path in artifacts or []:
                mlflow.log_artifact(str(path))
        return "mlflow"
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "experiment": experiment,
        "run": name,
        "time": datetime.now(timezone.utc).isoformat(),
        "params": {k: str(v) for k, v in params.items()},
        "metrics": {k: float(v) for k, v in metrics.items()},
        "artifacts": [str(p) for p in artifacts or []],
    }
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return str(path)


def read_runs(local_path: str | Path = "artifacts/runs.jsonl") -> list[dict[str, Any]]:
    path = Path(local_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
