"""Request counters in the Prometheus text exposition format.

Kept in process and without a client library: enough for a scraper to
follow request rates, error rates, latency and scored volume per endpoint.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.seconds: dict[str, float] = defaultdict(float)
        self.customers: dict[str, int] = defaultdict(int)

    def record(self, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(path, status)] += 1
            self.seconds[path] += seconds

    def add_customers(self, path: str, n: int) -> None:
        with self._lock:
            self.customers[path] += n

    def render(self) -> str:
        lines = [
            "# HELP loyalty_requests_total Requests by path and status code.",
            "# TYPE loyalty_requests_total counter",
        ]
        with self._lock:
            for (path, status), n in sorted(self.requests.items()):
                lines.append(
                    f'loyalty_requests_total{{path="{path}",status="{status}"}} {n}'
                )
            lines += [
                "# HELP loyalty_request_seconds_total Time spent handling requests.",
                "# TYPE loyalty_request_seconds_total counter",
            ]
            for path, sec in sorted(self.seconds.items()):
                lines.append(
                    f'loyalty_request_seconds_total{{path="{path}"}} {sec:.6f}'
                )
            lines += [
                "# HELP loyalty_customers_scored_total Customers scored by endpoint.",
                "# TYPE loyalty_customers_scored_total counter",
            ]
            for path, n in sorted(self.customers.items()):
                lines.append(f'loyalty_customers_scored_total{{path="{path}"}} {n}')
        return "\n".join(lines) + "\n"
