# Deployment

This note describes how a trained model moves from a laptop to a running scoring service and how it is watched once it is there. Everything is file based and runs without a database or a tracking server.

## 1. Train and register

```bash
loyalty-spend train --model ols --output artifacts/candidate.joblib
loyalty-spend model-card --model artifacts/candidate.joblib --output artifacts/MODEL_CARD.md
```

The bundle stores the fitted pipeline, its out-of-fold residuals (used for prediction intervals), the cross-validated metrics and a fingerprint of the training data. Register it and let it challenge the current production model on a labelled holdout:

```python
from loyalty_spend.models.persistence import load_bundle
from loyalty_spend.ops.registry import ModelRegistry, promote_if_better

registry = ModelRegistry("registry")
record = registry.register(load_bundle("artifacts/candidate.joblib"))
promoted, comparison = promote_if_better(
    registry, record.version, X_holdout, y_holdout, margin=0.01
)
```

The challenger replaces the champion only if the whole 95% bootstrap interval of the RMSE difference lies below `-margin`. The replaced version is archived, and `registry.rollback()` restores it.

## 2. Check every batch before scoring it

```bash
loyalty-spend monitor data/raw/test.csv --model artifacts/model.joblib --output artifacts/monitor.json
```

The command checks the batch against the scoring contract in `configs/contracts.yaml`, compares every feature with the training data (PSI with KS or chi-square tests under false discovery control), runs the domain classifier for joint shifts, and compares the predictions. It exits with status 0 for `ok` and `warn` and 2 for `alert`, so a pipeline can stop on an alert. Under an alert the conformal intervals are no longer guaranteed; `WeightedConformalRegressor` gives intervals that adapt to a covariate shift if scoring must go ahead.

## 3. Serve

```bash
make docker-serve       # builds the serve target and mounts ./artifacts
# or, without Docker:
pip install ".[serve]"
loyalty-spend serve --model artifacts/model.joblib --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | liveness check used by the container health check |
| `/version` | GET | package version, model name and training data fingerprint |
| `/predict` | POST | spend forecasts with `1 - alpha` conformal intervals |
| `/rewards` | POST | Monte Carlo reward cost of a batch: mean, 5% and 95% quantiles, expected shortfall |
| `/metrics` | GET | request, latency and volume counters in the Prometheus text format |

Requests are validated against the data dictionary; unknown levels, negative amounts and duplicated customer ids return 422, and batches larger than `LOYALTY_MAX_BATCH` (default 10,000) return 413. The container runs as a non-root user and reads the model from `LOYALTY_MODEL`.

## 4. Record what ran

`loyalty_spend.ops.manifest.build_manifest` records the package version, git commit, library versions, a hash of the configuration and a SHA-256 hash of every input file. `loyalty_spend.ops.tracking.log_run` logs parameters and metrics to MLflow when the `tracking` extra is installed and to `artifacts/runs.jsonl` otherwise.
