"""Dependency-free, process-local Prometheus metrics with bounded labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock

HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class LocalMetrics:
    """Collect low-cardinality runtime signals without external services."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._http_requests: Counter[tuple[str, str, str]] = Counter()
        self._http_duration_count: Counter[tuple[str, str]] = Counter()
        self._http_duration_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._http_duration_buckets: Counter[tuple[str, str, float]] = Counter()
        self._errors: Counter[tuple[str, str]] = Counter()
        self._ingestion_jobs: Counter[str] = Counter()
        self._dependencies: dict[str, int] = {}

    def observe_http(self, method: str, route: str, status_code: int, duration: float) -> None:
        """Record one completed request using route templates, never raw URLs."""
        method_label = method.upper()
        route_label = route if route.startswith("/") else "unmatched"
        status_class = f"{status_code // 100}xx"
        key = (method_label, route_label)
        with self._lock:
            self._http_requests[method_label, route_label, status_class] += 1
            self._http_duration_count[key] += 1
            self._http_duration_sum[key] += max(duration, 0)
            for bucket in HTTP_DURATION_BUCKETS:
                if duration <= bucket:
                    self._http_duration_buckets[method_label, route_label, bucket] += 1

    def observe_error(self, operation: str, error_type: str) -> None:
        """Count a sanitized error class at a bounded operation boundary."""
        with self._lock:
            self._errors[operation, error_type] += 1

    def observe_ingestion_job(self, outcome: str) -> None:
        """Count durable background job transitions by fixed outcome."""
        with self._lock:
            self._ingestion_jobs[outcome] += 1

    def set_dependency(self, dependency: str, available: bool) -> None:
        """Retain the latest local readiness observation."""
        with self._lock:
            self._dependencies[dependency] = int(available)

    def render_prometheus(self) -> str:
        """Render a deterministic Prometheus 0.0.4 text exposition."""
        with self._lock:
            requests = self._http_requests.copy()
            counts = self._http_duration_count.copy()
            sums = dict(self._http_duration_sum)
            buckets = self._http_duration_buckets.copy()
            errors = self._errors.copy()
            jobs = self._ingestion_jobs.copy()
            dependencies = dict(self._dependencies)

        lines = [
            "# HELP raglab_http_requests_total Completed HTTP requests.",
            "# TYPE raglab_http_requests_total counter",
        ]
        for (method, route, status_class), value in sorted(requests.items()):
            lines.append(
                "raglab_http_requests_total"
                f"{_labels(method=method, route=route, status_class=status_class)} {value}"
            )

        lines.extend(
            (
                "# HELP raglab_http_request_duration_seconds HTTP request duration.",
                "# TYPE raglab_http_request_duration_seconds histogram",
            )
        )
        for method, route in sorted(counts):
            for bucket in HTTP_DURATION_BUCKETS:
                value = buckets[method, route, bucket]
                lines.append(
                    "raglab_http_request_duration_seconds_bucket"
                    f"{_labels(method=method, route=route, le=_bucket_label(bucket))} {value}"
                )
            count = counts[method, route]
            lines.append(
                "raglab_http_request_duration_seconds_bucket"
                f"{_labels(method=method, route=route, le='+Inf')} {count}"
            )
            lines.append(
                "raglab_http_request_duration_seconds_sum"
                f"{_labels(method=method, route=route)} {sums[method, route]:.9f}"
            )
            lines.append(
                "raglab_http_request_duration_seconds_count"
                f"{_labels(method=method, route=route)} {count}"
            )

        lines.extend(
            (
                "# HELP raglab_errors_total Sanitized failures by operation and type.",
                "# TYPE raglab_errors_total counter",
            )
        )
        for (operation, error_type), value in sorted(errors.items()):
            lines.append(
                f"raglab_errors_total{_labels(operation=operation, error_type=error_type)} {value}"
            )

        lines.extend(
            (
                "# HELP raglab_ingestion_jobs_total Background ingestion outcomes.",
                "# TYPE raglab_ingestion_jobs_total counter",
            )
        )
        for outcome, value in sorted(jobs.items()):
            lines.append(f"raglab_ingestion_jobs_total{_labels(outcome=outcome)} {value}")

        lines.extend(
            (
                "# HELP raglab_dependency_up Latest readiness result for a local dependency.",
                "# TYPE raglab_dependency_up gauge",
            )
        )
        for dependency, value in sorted(dependencies.items()):
            lines.append(f"raglab_dependency_up{_labels(dependency=dependency)} {value}")
        return "\n".join(lines) + "\n"


def _labels(**values: str) -> str:
    rendered = ",".join(f'{key}="{_escape(value)}"' for key, value in values.items())
    return "{" + rendered + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _bucket_label(value: float) -> str:
    return f"{value:g}"
