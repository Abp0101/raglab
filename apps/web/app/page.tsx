"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Icon } from "@/components/icon";
import { PageIntro } from "@/components/page-intro";
import { SignalRibbon } from "@/components/signal-ribbon";
import { StateNote } from "@/components/state-note";
import { apiFetch, DEMO_MODE } from "@/lib/api";
import { demoCollections, demoPipelines, demoResponse } from "@/lib/demo-data";
import { formatDate, formatLatency, titleCase } from "@/lib/format";
import type { Collection, CursorPage, HealthReadiness, PipelineSummary } from "@/types/api";

export default function OverviewPage() {
  const [collections, setCollections] = useState<Collection[]>(DEMO_MODE ? demoCollections : []);
  const [pipelines, setPipelines] = useState<PipelineSummary[]>(DEMO_MODE ? demoPipelines : []);
  const [readiness, setReadiness] = useState<HealthReadiness | null>(
    DEMO_MODE ? { status: "ok", dependencies: { postgres: true, qdrant: true, redis: true } } : null,
  );
  const [preview, setPreview] = useState(DEMO_MODE);

  useEffect(() => {
    if (DEMO_MODE) return;
    Promise.all([
      apiFetch<CursorPage<Collection>>("/collections?limit=4"),
      apiFetch<PipelineSummary[]>("/pipelines"),
      apiFetch<HealthReadiness>("/health/ready"),
    ])
      .then(([collectionPage, pipelineList, health]) => {
        setCollections(collectionPage.items);
        setPipelines(pipelineList);
        setReadiness(health);
      })
      .catch(() => {
        setCollections(demoCollections);
        setPipelines(demoPipelines);
        setReadiness({ status: "degraded", dependencies: { postgres: false, qdrant: false, redis: false } });
        setPreview(true);
      });
  }, []);

  const available = pipelines.filter((pipeline) => pipeline.available).length;
  const documentCount = collections.reduce((sum, collection) => sum + collection.document_count, 0);

  return (
    <div className="page-stack overview-page">
      <PageIntro
        index="01"
        kicker="LOCAL RETRIEVAL INSTRUMENT"
        accessibleTitle="Interrogate the retrieval, not just the answer."
        title={<>Interrogate the retrieval,<br /><em>not just the answer.</em></>}
        description="One workbench for tracing evidence, comparing five orchestration paths, and measuring what the model was actually allowed to see."
        aside={<Link className="button primary" href="/query">Open evidence run <Icon name="arrow" /></Link>}
      />

      {preview && (
        <StateNote title="Interface preview" tone="neutral">
          The local API is not connected, so this view uses clearly labelled biomedical sample data. Live actions remain disabled until RAGLab is online.
        </StateNote>
      )}

      <SignalRibbon
        signals={[
          { label: "PIPELINES READY", value: `${available.toString().padStart(2, "0")} / 05`, tone: available === 5 ? "good" : "warn" },
          { label: "INDEXED DOCUMENTS", value: documentCount.toString().padStart(3, "0") },
          { label: "LAST TRACE", value: formatLatency(demoResponse.latency.total_ms), tone: "accent" },
          { label: "PAID API COST", value: "$0.00", tone: "good" },
          { label: "DATA PLANE", value: readiness?.status === "ok" ? "NOMINAL" : "DEGRADED", tone: readiness?.status === "ok" ? "good" : "warn" },
        ]}
      />

      <div className="overview-grid">
        <section className="ruled-panel corpus-ledger">
          <div className="section-heading">
            <div><span className="eyebrow">CORPUS LEDGER</span><h2>Shared evidence collections</h2></div>
            <Link className="text-link" href="/library">Manage library <Icon name="arrow" /></Link>
          </div>
          <div className="ledger-list">
            {collections.map((collection, index) => (
              <Link className="ledger-row" href={`/library?collection=${collection.collection_id}`} key={collection.collection_id}>
                <span className="ledger-number">{String(index + 1).padStart(2, "0")}</span>
                <div><strong>{collection.name}</strong><p>{collection.description ?? "No collection note"}</p></div>
                <div className="ledger-meta"><strong>{collection.document_count}</strong><span>documents</span></div>
                <span className="ledger-date">{formatDate(collection.updated_at)}</span>
              </Link>
            ))}
          </div>
        </section>

        <section className="dark-panel pipeline-array">
          <div className="section-heading inverse">
            <div><span className="eyebrow">ORCHESTRATION ARRAY</span><h2>Five paths. One contract.</h2></div>
            <span className="array-tag">SHARED DATA PLANE</span>
          </div>
          <div className="pipeline-lines">
            {pipelines.map((pipeline, index) => (
              <div className="pipeline-line" key={pipeline.framework}>
                <span className="pipeline-index">0{index + 1}</span>
                <div className="pipeline-name"><strong>{titleCase(pipeline.framework)}</strong><span>{pipeline.capabilities.agentic ? "bounded graph" : "native adapter"}</span></div>
                <div className="capability-dots" aria-label={`${pipeline.framework} capabilities`}>
                  {[pipeline.capabilities.hybrid_retrieval, pipeline.capabilities.reranking, pipeline.capabilities.streaming].map((enabled, dot) => <i className={enabled ? "is-on" : ""} key={dot} />)}
                </div>
                <span className={`availability ${pipeline.available ? "is-ready" : ""}`}>{pipeline.available ? "READY" : "PLANNED"}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="trace-feature">
        <div className="trace-copy">
          <span className="eyebrow">REFERENCE TRACE / LANGGRAPH</span>
          <h2>Every answer leaves an evidence trail.</h2>
          <p>{demoResponse.answer}</p>
          <div className="trace-actions"><Link className="button secondary" href="/query">Inspect ranked chunks</Link><span>CONFIDENCE {Math.round((demoResponse.confidence ?? 0) * 100)}%</span></div>
        </div>
        <div className="trace-map" aria-label="Retrieval trace from query to grounded answer">
          <div className="trace-node input"><span>01</span><strong>QUERY</strong><small>sampling rate?</small></div>
          <div className="trace-connector"><i /><i /><i /></div>
          <div className="trace-node retrieval"><span>02</span><strong>HYBRID</strong><small>dense + BM25</small></div>
          <div className="trace-connector"><i /><i /></div>
          <div className="trace-node rerank"><span>03</span><strong>RERANK</strong><small>cross-encoder</small></div>
          <div className="trace-connector"><i /></div>
          <div className="trace-node answer"><span>04</span><strong>GROUNDED</strong><small>1 citation</small></div>
        </div>
      </section>
    </div>
  );
}
