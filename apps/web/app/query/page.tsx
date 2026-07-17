"use client";

import { useEffect, useMemo, useState } from "react";

import { Icon } from "@/components/icon";
import { PageIntro } from "@/components/page-intro";
import { SignalRibbon } from "@/components/signal-ribbon";
import { API_URL, apiFetch, DEMO_MODE, errorMessage, storedApiKey } from "@/lib/api";
import { demoCollections, demoPipelines, demoResponse } from "@/lib/demo-data";
import { evidenceSignalFor, relativeScorePercent } from "@/lib/evidence-score";
import { formatLatency, formatNumber, shortId, titleCase } from "@/lib/format";
import { drainSseBuffer } from "@/lib/sse";
import type { Collection, CursorPage, Framework, PipelineSummary, QueryPayload, RAGResponse, RetrievalMode } from "@/types/api";

type RunState = "idle" | "accepted" | "processing" | "complete" | "failed";

const frameworkCodes: Record<Framework, string> = {
  custom: "PY",
  langchain: "LC",
  langgraph: "LG",
  llamaindex: "LI",
  haystack: "HS",
};

export default function QueryPage() {
  const [collections, setCollections] = useState<Collection[]>(demoCollections);
  const [pipelines, setPipelines] = useState<PipelineSummary[]>(demoPipelines);
  const [collectionId, setCollectionId] = useState(demoCollections[0].collection_id);
  const [framework, setFramework] = useState<Framework>("langgraph");
  const [question, setQuestion] = useState("What sampling rate was used for the wearable IMUs, and how were the sensors synchronized?");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [rerank, setRerank] = useState(true);
  const [runState, setRunState] = useState<RunState>("idle");
  const [response, setResponse] = useState<RAGResponse>(demoResponse);
  const [selectedChunk, setSelectedChunk] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isReference, setIsReference] = useState(true);

  useEffect(() => {
    if (DEMO_MODE) return;
    Promise.all([
      apiFetch<CursorPage<Collection>>("/collections?limit=100"),
      apiFetch<PipelineSummary[]>("/pipelines"),
    ])
      .then(([page, list]) => {
        if (page.items.length) {
          setCollections(page.items);
          setCollectionId(page.items[0].collection_id);
        }
        setPipelines(list);
      })
      .catch(() => {
        // The reference trace remains visible and explicitly labelled.
      });
  }, []);

  const selected = response.retrieved_chunks[selectedChunk] ?? response.retrieved_chunks[0];
  const evidenceSignal = evidenceSignalFor(selected);
  const availableFrameworks = useMemo(() => new Set(pipelines.filter((item) => item.available).map((item) => item.framework)), [pipelines]);

  const runPreview = async () => {
    setRunState("accepted");
    await new Promise((resolve) => window.setTimeout(resolve, 280));
    setRunState("processing");
    await new Promise((resolve) => window.setTimeout(resolve, 620));
    setResponse({ ...demoResponse, framework });
    setSelectedChunk(0);
    setIsReference(true);
    setRunState("complete");
  };

  const execute = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    if (DEMO_MODE) return runPreview();
    const payload: QueryPayload = {
      query: question,
      framework,
      collection_id: collectionId,
      top_k: topK,
      retrieval_mode: retrievalMode,
      rerank,
      temperature: 0,
      debug: true,
    };
    setRunState("accepted");
    try {
      const headers = new Headers({ "Content-Type": "application/json", Accept: "text/event-stream" });
      const key = storedApiKey();
      if (key) headers.set("Authorization", `Bearer ${key}`);
      const stream = await fetch(`${API_URL}/query/stream`, { method: "POST", headers, body: JSON.stringify(payload) });
      if (!stream.ok || !stream.body) throw new Error(`Query stream returned HTTP ${stream.status}`);
      const reader = stream.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const drained = drainSseBuffer(buffer);
        buffer = drained.remainder;
        for (const message of drained.events) {
          if (message.event === "query.accepted" || message.event === "query.heartbeat") setRunState("processing");
          if (message.event === "query.result") {
            setResponse(message.data as RAGResponse);
            setSelectedChunk(0);
            setIsReference(false);
            setRunState("complete");
          }
          if (message.event === "query.error") {
            const envelope = message.data as { error?: { message?: string } };
            throw new Error(envelope.error?.message ?? "The local pipeline failed.");
          }
        }
        if (done) break;
      }
    } catch (caught) {
      setError(errorMessage(caught));
      setRunState("failed");
    }
  };

  return (
    <div className="page-stack query-page">
      <PageIntro
        index="02"
        kicker="QUERY / RETRIEVE / VERIFY"
        accessibleTitle="Evidence workbench"
        title={<>Evidence<br /><em>workbench.</em></>}
        description="Run one shared query contract, then inspect the exact chunks, scores, citations, and timing behind the response."
        aside={<div className="run-serial"><span>TRACE</span><strong>{isReference ? "REFERENCE / 001" : shortId(response.retrieved_chunks[0]?.chunk.chunk_id ?? "LIVE")}</strong></div>}
      />

      <form className="query-console" onSubmit={execute}>
        <section className="query-controls" aria-label="Query controls">
          <div className="console-heading"><span className="eyebrow">INPUT CHANNEL</span><strong>01 / Configure</strong></div>
          <label className="field-label" htmlFor="collection">Evidence collection</label>
          <select className="select-input" id="collection" value={collectionId} onChange={(event) => setCollectionId(event.target.value)}>
            {collections.map((collection) => <option value={collection.collection_id} key={collection.collection_id}>{collection.name}</option>)}
          </select>

          <fieldset className="framework-fieldset">
            <legend className="field-label">Framework path</legend>
            <div className="framework-switch">
              {demoPipelines.map((pipeline, index) => (
                <button type="button" key={pipeline.framework} className="framework-key" data-active={framework === pipeline.framework || undefined} disabled={!availableFrameworks.has(pipeline.framework)} onClick={() => setFramework(pipeline.framework)} aria-pressed={framework === pipeline.framework} aria-label={`Select ${titleCase(pipeline.framework)} framework`}>
                  <span>0{index + 1}</span><strong>{frameworkCodes[pipeline.framework]}</strong><small>{titleCase(pipeline.framework)}</small>
                </button>
              ))}
            </div>
          </fieldset>

          <label className="field-label" htmlFor="question">Research question</label>
          <textarea className="query-textarea" id="question" value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={4000} required />
          <div className="character-count"><span>Grounded answer required</span><span>{question.length} / 4000</span></div>

          <div className="control-grid">
            <label><span className="field-label">Retrieval</span><select className="select-input" value={retrievalMode} onChange={(event) => setRetrievalMode(event.target.value as RetrievalMode)}><option value="hybrid">Hybrid / RRF</option><option value="dense">Dense</option><option value="sparse">BM25 sparse</option></select></label>
            <label><span className="field-label">Final K</span><input className="number-input" type="number" min="1" max="20" value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></label>
          </div>
          <label className="switch-row"><span><strong>Cross-encoder reranking</strong><small>Local model, no external calls</small></span><input type="checkbox" checked={rerank} onChange={(event) => setRerank(event.target.checked)} /><i aria-hidden="true" /></label>
          <button className="run-button" type="submit" disabled={runState === "accepted" || runState === "processing"}>
            <span>{runState === "processing" ? "Pipeline running" : "Run evidence trace"}</span><Icon name="arrow" />
          </button>
          {error && <div className="inline-error" role="alert"><Icon name="warning" /><span>{error}</span></div>}
        </section>

        <section className="answer-stage" aria-label="Grounded response" aria-live="polite">
          <div className="console-heading"><span className="eyebrow">SYNTHESIS CHANNEL</span><strong>02 / Grounded response</strong><span className={`run-state state-${runState}`}>{runState === "idle" ? "REFERENCE" : runState.toUpperCase()}</span></div>
          <SignalRibbon signals={[
            { label: "FRAMEWORK", value: titleCase(response.framework), tone: "accent" },
            { label: "EVIDENCE", value: response.evidence_status.toUpperCase(), tone: response.evidence_status === "sufficient" ? "good" : "warn" },
            { label: "CONFIDENCE", value: response.confidence == null ? "—" : `${Math.round(response.confidence * 100)}%` },
            { label: "LATENCY", value: formatLatency(response.latency.total_ms) },
            { label: "COST", value: `$${(response.usage.estimated_cost_usd ?? 0).toFixed(2)}`, tone: "good" },
          ]} />
          <article className="answer-document">
            <div className="answer-margin"><span>A</span><span>{response.citations.length.toString().padStart(2, "0")} cites</span></div>
            <div className="answer-body">
              <span className="document-label">VALIDATED ANSWER / {response.model}</span>
              <p>{response.answer}</p>
              <div className="citation-index">
                {response.citations.map((citation, index) => (
                  <button type="button" key={citation.chunk_id} onClick={() => {
                    const match = response.retrieved_chunks.findIndex((item) => item.chunk.chunk_id === citation.chunk_id);
                    if (match >= 0) setSelectedChunk(match);
                  }}><span>[{index + 1}]</span>{citation.document_title}<small>p. {citation.page_number ?? "—"} / {citation.section_heading ?? "unsectioned"}</small></button>
                ))}
              </div>
            </div>
          </article>
          <div className="latency-strip" aria-label="Query stage latency">
            {(["retrieval_ms", "reranking_ms", "generation_ms"] as const).map((key) => {
              const value = response.latency[key];
              const width = Math.max(4, (value / response.latency.total_ms) * 100);
              return <div key={key}><span>{key.replace("_ms", "")}</span><i style={{ width: `${width}%` }} /><strong>{formatLatency(value)}</strong></div>;
            })}
          </div>
        </section>

        <aside className="evidence-rail" aria-label="Retrieved evidence">
          <div className="console-heading"><span className="eyebrow">EVIDENCE CHANNEL</span><strong>03 / Ranked chunks</strong><span>{response.retrieved_chunks.length} ITEMS</span></div>
          <div className="evidence-list">
            {response.retrieved_chunks.map((item, index) => (
              <button type="button" className="evidence-item" data-active={selectedChunk === index || undefined} key={item.chunk.chunk_id} onClick={() => setSelectedChunk(index)} aria-pressed={selectedChunk === index}>
                <span className="rank">{String(item.rank).padStart(2, "0")}</span>
                <div className="evidence-item-copy"><strong>{item.chunk.metadata.display_title}</strong><span>p. {item.chunk.metadata.page_number ?? "—"} / {item.chunk.metadata.section_heading ?? "unsectioned"}</span><p>{item.chunk.text}</p></div>
                <div className="score-gauge"><i style={{ height: `${relativeScorePercent(item, response.retrieved_chunks)}%` }} /><span>{formatNumber(item.reranker_score ?? item.relevance_score, 3)}</span></div>
              </button>
            ))}
          </div>
          {selected && (
            <div className="evidence-inspector">
              <div className="inspector-grid"><span>dense<strong>{formatNumber(selected.dense_score, 3)}</strong></span><span>sparse<strong>{formatNumber(selected.sparse_score, 3)}</strong></span><span>fusion<strong>{formatNumber(selected.fusion_score, 4)}</strong></span><span>rerank<strong>{formatNumber(selected.reranker_score, 3)}</strong></span></div>
              <blockquote>{selected.chunk.text}</blockquote>
              <div className="provenance"><span>CHUNK {shortId(selected.chunk.chunk_id)}</span><span>{selected.chunk.token_count ?? "—"} TOKENS</span></div>
            </div>
          )}
          <div className="evidence-score" aria-label="Selected evidence score">
            <span>{evidenceSignal.label}</span>
            <strong>{formatNumber(evidenceSignal.value, evidenceSignal.digits)}</strong>
          </div>
        </aside>
      </form>
    </div>
  );
}
