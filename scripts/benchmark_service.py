"""Time the scoring service in process for several batch sizes.

Usage::

    python scripts/benchmark_service.py artifacts/model.joblib
"""

from __future__ import annotations

import argparse

from fastapi.testclient import TestClient

from loyalty_spend.serving.api import create_app
from loyalty_spend.serving.benchmark import benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    client = TestClient(create_app(args.model))
    table = benchmark(client, args.sizes, args.repeats)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
