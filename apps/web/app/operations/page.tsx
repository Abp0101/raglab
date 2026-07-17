"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { Icon } from "@/components/icon";
import { PageIntro } from "@/components/page-intro";
import { SignalRibbon } from "@/components/signal-ribbon";
import { apiFetch, apiText, DEMO_MODE, errorMessage } from "@/lib/api";
import { demoMetrics } from "@/lib/demo-data";
import { parsePrometheus, sumMetric } from "@/lib/metrics";
import type { HealthReadiness } from "@/types/api";

const demoHealth: HealthReadiness = { status: "ok", dependencies: { postgres: true, qdrant: true, redis: true } };

export default function OperationsPage() {
  const [health, setHealth] = useState<HealthReadiness>(demoHealth);
  const [rawMetrics, setRawMetrics] = useState(demoMetrics);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState(DEMO_MODE);

  const refresh = useCallback(async () => {
    if (DEMO_MODE) {
      setLastChecked(new Date());
      return;
    }
    try {
      const [ready, metrics] = await Promise.all([
        apiFetch<HealthReadiness>("/health/ready"),
        apiText("/metrics"),
      ]);
      setHealth(ready);
      setRawMetrics(metrics);
      setPreview(false);
      setError(null);
      setLastChecked(new Date());
    } catch (caught) {
      setError(errorMessage(caught));
      setPreview(true);
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 15_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  const samples = useMemo(() => parsePrometheus(rawMetrics), [rawMetrics]);
  const requests = sumMetric(samples, "raglab_http_requests_total");
  const errors = sumMetric(samples, "raglab_errors_total");
  const completedJobs = samples.find((sample) => sample.name === "raglab_ingestion_jobs_total" && sample.labels.outcome === "completed")?.value ?? 0;
  const requestSamples = samples.filter((sample) => sample.name === "raglab_http_requests_total").sort((a, b) => b.value - a.value);
  const maxRequests = Math.max(...requestSamples.map((sample) => sample.value), 1);

  return (
    <div className="page-stack operations-page">
      <PageIntro index="05" kicker="LOCAL SIGNALS / NO OUTBOUND EXPORTER" accessibleTitle="Operations scope" title={<>Operations<br /><em>scope.</em></>} description="Read bounded process metrics, isolate degraded dependencies, and follow a failure from request ID to durable job state." aside={<button className="button primary" type="button" onClick={() => void refresh()}>Refresh signals <Icon name="arrow" /></button>} />

      {error && <div className="notice-bar warning" role="alert"><span>Live API unavailable: {error}. Showing the reference signal set.</span></div>}

      <SignalRibbon signals={[
        { label: "READINESS", value: health.status.toUpperCase(), tone: health.status === "ok" ? "good" : "warn" },
        { label: "REQUESTS OBSERVED", value: requests.toFixed(0).padStart(3, "0") },
        { label: "ERROR SIGNALS", value: errors.toFixed(0).padStart(2, "0"), tone: errors ? "warn" : "good" },
        { label: "JOBS COMPLETED", value: completedJobs.toFixed(0).padStart(2, "0") },
        { label: "REGISTRY", value: preview ? "REFERENCE" : "PROCESS LOCAL", tone: "accent" },
      ]} />

      <div className="operations-grid">
        <section className="dependency-panel dark-panel">
          <div className="section-heading inverse"><div><span className="eyebrow">DEPENDENCY ARRAY</span><h2>Local data plane</h2></div><span className="array-tag">{lastChecked ? lastChecked.toLocaleTimeString("en-GB") : "CHECKING"}</span></div>
          <div className="dependency-list">
            {Object.entries(health.dependencies).map(([name, available], index) => (
              <div className="dependency-row" key={name}><span className="dependency-index">0{index + 1}</span><div><strong>{name.toUpperCase()}</strong><small>{name === "postgres" ? "catalog + job leases" : name === "qdrant" ? "dense vector plane" : "BM25 sparse plane"}</small></div><div className={`dependency-signal ${available ? "is-up" : "is-down"}`}><i /><span>{available ? "RESPONDING" : "UNAVAILABLE"}</span></div></div>
            ))}
          </div>
          <div className="local-only-stamp"><span>TELEMETRY BOUNDARY</span><strong>THIS PROCESS ONLY</strong><p>No analytics client, hosted collector, credential, or paid endpoint is constructed.</p></div>
        </section>

        <section className="traffic-panel">
          <div className="section-heading"><div><span className="eyebrow">ROUTE TEMPLATES</span><h2>Observed request volume</h2></div><span className="source-label">BOUNDED LABELS</span></div>
          <div className="route-bars">
            {requestSamples.map((sample) => <div className="route-bar" key={`${sample.labels.method}-${sample.labels.route}-${sample.labels.status_class}`}><span>{sample.labels.method}</span><strong>{sample.labels.route}</strong><div><i style={{ width: `${(sample.value / maxRequests) * 100}%` }} /></div><span>{sample.labels.status_class}</span><b>{sample.value}</b></div>)}
          </div>
        </section>
      </div>

      <div className="operations-lower-grid">
        <section className="error-register">
          <div className="section-heading"><div><span className="eyebrow">SANITIZED FAILURE REGISTER</span><h2>Error types</h2></div><span className="array-tag">NO REQUEST CONTENT</span></div>
          <div className="error-table"><div className="error-head"><span>Operation</span><span>Type</span><span>Count</span></div>{samples.filter((sample) => sample.name === "raglab_errors_total").map((sample) => <div className="error-row" key={`${sample.labels.operation}-${sample.labels.error_type}`}><strong>{sample.labels.operation}</strong><span>{sample.labels.error_type}</span><b>{sample.value}</b></div>)}</div>
        </section>
        <aside className="runbook-card">
          <span className="eyebrow">FAILURE RUNBOOK / 04 STEPS</span><h2>Trace before retry.</h2>
          <ol><li><span>01</span><p>Confirm the API process at <code>/health/live</code>.</p></li><li><span>02</span><p>Isolate PostgreSQL, Qdrant, or Redis at <code>/health/ready</code>.</p></li><li><span>03</span><p>Match the safe error type to a structured request or job log.</p></li><li><span>04</span><p>Retry HTTP 503 only after the dependency responds.</p></li></ol>
        </aside>
      </div>
    </div>
  );
}
