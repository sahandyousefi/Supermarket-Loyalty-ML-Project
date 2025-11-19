"""Latency of the scoring endpoint by batch size.

Requests go through the full application stack in process (validation,
cleaning, the model and serialisation) with FastAPI's test client, so the
figures exclude network time and measure what the service itself adds.
Each batch size is timed over several requests after a warm-up request,
and the median and 95th percentile per request and per customer are
reported.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

REGIONS = ["Americas", "Asia/Pacific", "Europe", "Middle East/Africa"]
TENURE = ["0-1", "1-3", "3-5", "5-10", "10+"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def synthetic_customers(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [
        {
            "customer_id": i,
            "first_month": float(round(rng.uniform(10, 60), 2)),
            "items_in_first_month": int(rng.integers(5, 40)),
            "region": REGIONS[i % 4],
            "loyalty_years": TENURE[i % 5],
            "joining_month": MONTHS[i % 6],
            "promotion": "Yes" if i % 2 else "No",
        }
        for i in range(n)
    ]


def benchmark(
    client: Any,
    sizes: Sequence[int] = (1, 10, 100, 1000),
    repeats: int = 20,
    endpoint: str = "/predict",
) -> pd.DataFrame:
    rows = []
    for size in sizes:
        payload = {"customers": synthetic_customers(size)}
        client.post(endpoint, json=payload).raise_for_status()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            client.post(endpoint, json=payload).raise_for_status()
            times.append(time.perf_counter() - start)
        ms = 1000 * np.asarray(times)
        rows.append(
            {
                "batch_size": size,
                "median_ms": float(np.median(ms)),
                "p95_ms": float(np.quantile(ms, 0.95)),
                "median_ms_per_customer": float(np.median(ms) / size),
            }
        )
    return pd.DataFrame(rows)
