from raglab.core.metrics import LocalMetrics


def test_local_metrics_render_prometheus_counters_histograms_and_gauges() -> None:
    metrics = LocalMetrics()
    metrics.observe_http("get", "/documents/{document_id}", 200, 0.012)
    metrics.observe_http("GET", "/documents/{document_id}", 503, 0.3)
    metrics.observe_error("readiness", "TimeoutError")
    metrics.observe_ingestion_job("failed")
    metrics.set_dependency("postgres", False)

    rendered = metrics.render_prometheus()

    assert (
        'raglab_http_requests_total{method="GET",route="/documents/{document_id}",'
        'status_class="2xx"} 1'
    ) in rendered
    assert (
        'raglab_http_requests_total{method="GET",route="/documents/{document_id}",'
        'status_class="5xx"} 1'
    ) in rendered
    assert (
        'raglab_http_request_duration_seconds_count{method="GET",'
        'route="/documents/{document_id}"} 2'
    ) in rendered
    assert 'raglab_errors_total{operation="readiness",error_type="TimeoutError"} 1' in rendered
    assert 'raglab_ingestion_jobs_total{outcome="failed"} 1' in rendered
    assert 'raglab_dependency_up{dependency="postgres"} 0' in rendered


def test_local_metrics_use_unmatched_instead_of_a_raw_non_route_label() -> None:
    metrics = LocalMetrics()

    metrics.observe_http("GET", "user-supplied-value", 404, 0)

    rendered = metrics.render_prometheus()
    assert 'route="unmatched"' in rendered
    assert "user-supplied-value" not in rendered
