# Deployment

This note describes how a trained model moves from a laptop to a running scoring service and how it is watched once it is there. Everything is file based and runs without a database or a tracking server.

## 1. Train and register

```bash
loyalty-spend train --output artifacts/candidate.joblib     # wls_month, the champion
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

The challenger replaces the champion only if the whole 95% bootstrap interval of the RMSE difference lies below `-margin`. The replaced version is archived, and `registry.rollback()` restores it. `loyalty_spend.ops.champion.promotion_decision` adds a second route for a challenger that is as accurate but gives better intervals: its RMSE may not be worse by more than a non-inferiority margin and its interval score must be better beyond sampling error. This is how the month-weighted model replaced OLS.

A bundle of the weighted model also stores out-of-fold residuals divided by each member's error SD. Batch scoring and the service then give normalized conformal intervals, whose width follows the member's joining month.

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

In CI, a model trained on the runner is mounted read-only into a freshly built service image, and the job waits for `/health` and then calls `/version` and `/predict`, so a broken image or start command fails the build rather than the first deployment.

Every response carries an `X-Request-ID` header, echoed from the request when the caller sends a printable identifier of up to 128 characters and generated otherwise. Each request is logged as one JSON line with the identifier, method, path, status, latency in milliseconds, model name and package version.

**Latency.** `python scripts/benchmark_service.py artifacts/model.joblib` times `/predict` in process, through validation, cleaning, the model and serialisation but without network time. For the OLS champion on one CPU core, 20 requests per batch size after a warm-up:

| Customers per request | Median (ms) | 95th percentile (ms) | Median per customer (ms) |
| --- | --- | --- | --- |
| 1 | 34.3 | 48.6 | 34.3 |
| 10 | 30.6 | 39.5 | 3.1 |
| 100 | 34.6 | 49.7 | 0.35 |
| 1,000 | 54.7 | 80.5 | 0.055 |

A request costs about 30 ms whatever its size, mostly fixed work in validation and the preprocessing pipeline, so callers with many members should send them in batches of hundreds rather than one at a time.

## 4. Record what ran

`loyalty_spend.ops.manifest.build_manifest` records the package version, git commit, library versions, a hash of the configuration and a SHA-256 hash of every input file. `loyalty_spend.ops.tracking.log_run` logs parameters and metrics to MLflow when the `tracking` extra is installed and to `artifacts/runs.jsonl` otherwise.

## 5. Check that the results reproduce

`configs/reference.yaml` lists 17 reported results (driver statistics, promotion estimates, reward cost and tail risk of the programme design, WAIC of the Bayesian models) with the value obtained when the reference was made and a relative tolerance. Closed-form statistics have tolerances of $10^{-6}$; Monte Carlo and Gibbs results, which depend on the order of floating point operations in the linear algebra library, have $10^{-3}$ to $2 \times 10^{-3}$.

```bash
loyalty-spend reproduce --output artifacts/reproduce.json   # exit code 1 if any result moved
loyalty-spend reproduce --update                            # after an intended change
```

The record written by `--output` holds the comparison, the manifest of section 4 and the BLAS and OpenMP libraries in use with their thread counts. A separate test runs the out-of-fold forecasts, a conformal quantile, the reward simulation and the synthetic sampler in fresh processes with different hash seeds and one or two threads: everything agrees exactly except least squares, which differs by less than $10^{-9}$ because more threads sum in a different order.
