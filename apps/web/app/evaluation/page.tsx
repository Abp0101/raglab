"use client";

import { useEffect, useMemo, useState } from "react";

import { PageIntro } from "@/components/page-intro";
import { SignalRibbon } from "@/components/signal-ribbon";
import { formatNumber } from "@/lib/format";
import type { BenchmarkReport } from "@/types/api";

const fallbackReport: BenchmarkReport = {
  title: "Five-framework local baseline",
  runDate: "2026-07-17",
  model: "local llama3.2:latest",
  cost: "$0.00",
  frameworks: ["Custom", "LangChain", "LangGraph", "LlamaIndex", "Haystack"],
  metrics: [
    { name: "Citation precision", values: { Custom: 0.8056, LangChain: 0.8333, LangGraph: 1, LlamaIndex: 1, Haystack: 0.8333 } },
    { name: "Citation recall", values: { Custom: 1, LangChain: 0.8333, LangGraph: 1, LlamaIndex: 1, Haystack: 0.8333 } },
    { name: "Key-fact coverage", values: { Custom: 0.4167, LangChain: 0.4167, LangGraph: 0.4167, LlamaIndex: 0.4167, Haystack: 0.4167 } },
    { name: "Mean latency (ms)", values: { Custom: 2726.7392, LangChain: 1111.8233, LangGraph: 1332.2198, LlamaIndex: 1891.7938, Haystack: 1508.1083 } },
    { name: "MRR", values: { Custom: 1, LangChain: 1, LangGraph: 1, LlamaIndex: 1, Haystack: 1 } },
    { name: "NDCG", values: { Custom: 1, LangChain: 1, LangGraph: 1, LlamaIndex: 1, Haystack: 1 } },
    { name: "Refusal accuracy", values: { Custom: 0.8571, LangChain: 0.7143, LangGraph: 1, LlamaIndex: 0.8571, Haystack: 0.7143 } },
    { name: "Retrieval recall", values: { Custom: 1, LangChain: 1, LangGraph: 1, LlamaIndex: 1, Haystack: 1 } },
  ],
  source: "reports/baselines/custom-vs-langchain-vs-langgraph-vs-llamaindex-vs-haystack-llama3.2-v1.md",
};

const frameworkCodes = ["PY", "LC", "LG", "LI", "HS"];
const nativeIndexing = [
  { framework: "Custom", chunks: 2.33, containment: 1, recall: 0.833, latency: 0.18 },
  { framework: "LangChain", chunks: 2.67, containment: 1, recall: 0.667, latency: 0.16 },
  { framework: "LlamaIndex", chunks: 2, containment: 1, recall: 0.833, latency: 15.99 },
  { framework: "Haystack", chunks: 2, containment: 1, recall: 0.833, latency: 0.39 },
];

export default function EvaluationPage() {
  const [report, setReport] = useState<BenchmarkReport>(fallbackReport);
  const [sourceState, setSourceState] = useState<"committed" | "reference">("reference");
  const [focusMetric, setFocusMetric] = useState("Citation precision");

  useEffect(() => {
    fetch("/api/reports", { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("baseline unavailable");
        return (await response.json()) as BenchmarkReport;
      })
      .then((loaded) => {
        setReport(loaded);
        setSourceState("committed");
      })
      .catch(() => setSourceState("reference"));
  }, []);

  const focused = report.metrics.find((metric) => metric.name === focusMetric) ?? report.metrics[0];
  const maxFocused = Math.max(...Object.values(focused?.values ?? { fallback: 1 }));
  const displayMetrics = useMemo(
    () => report.metrics.filter((metric) => ["Citation precision", "Citation recall", "Key-fact coverage", "Mean latency (ms)", "MRR", "NDCG", "Refusal accuracy", "Retrieval recall"].includes(metric.name)),
    [report],
  );

  return (
    <div className="page-stack evaluation-page">
      <PageIntro index="04" kicker="CONTROLLED OBSERVATION / NOT A LEADERBOARD" accessibleTitle="Framework evidence matrix" title={<>Framework<br /><em>evidence matrix.</em></>} description="Compare five orchestration paths over the same corpus, retrieval service, local model, question order, and evaluation contract." aside={<div className="run-serial"><span>BASELINE SOURCE</span><strong>{sourceState === "committed" ? "COMMITTED / LIVE" : "REFERENCE / EMBEDDED"}</strong></div>} />

      <SignalRibbon signals={[
        { label: "DATASET", value: "v1.0.0" },
        { label: "QUESTIONS", value: "07 / 07", tone: "good" },
        { label: "MODEL", value: report.model.replace("local ", "") },
        { label: "RUN DATE", value: report.runDate },
        { label: "PAID API COST", value: report.cost, tone: "good" },
      ]} />

      <section className="metric-focus dark-panel">
        <div className="section-heading inverse"><div><span className="eyebrow">MEASUREMENT LENS</span><h2>{focused?.name}</h2></div><select className="dark-select" aria-label="Focus metric" value={focusMetric} onChange={(event) => setFocusMetric(event.target.value)}>{displayMetrics.map((metric) => <option key={metric.name}>{metric.name}</option>)}</select></div>
        <div className="framework-lanes">
          {report.frameworks.map((framework, index) => {
            const value = focused?.values[framework] ?? 0;
            return <div className={`framework-lane framework-${index}`} key={framework}><div className="lane-label"><span>{frameworkCodes[index]}</span><strong>{framework}</strong></div><div className="lane-track"><i style={{ width: `${Math.max(2, (value / maxFocused) * 100)}%` }} /></div><strong className="lane-value">{focusMetric.includes("latency") ? `${value.toFixed(0)} ms` : formatNumber(value, 4)}</strong></div>;
          })}
        </div>
        <p className="measurement-note">Bars are normalized only within the selected measurement. Latency is a single local observation; taller does not mean better.</p>
      </section>

      <section className="matrix-panel">
        <div className="section-heading"><div><span className="eyebrow">OBSERVED AGGREGATES</span><h2>Shared-pipeline comparison</h2></div><span className="source-label">SOURCE / {report.source.split("/").at(-1)}</span></div>
        <div className="measurement-table-wrap">
          <table className="measurement-table">
            <thead><tr><th scope="col">Metric</th>{report.frameworks.map((framework, index) => <th scope="col" key={framework}><span>0{index + 1}</span>{framework}</th>)}</tr></thead>
            <tbody>{displayMetrics.map((metric) => {
              const maximum = Math.max(...Object.values(metric.values));
              return <tr key={metric.name}><th scope="row">{metric.name}</th>{report.frameworks.map((framework) => { const value = metric.values[framework] ?? 0; return <td key={framework}><i style={{ width: `${Math.max(2, (value / maximum) * 100)}%` }} /><span>{metric.name.includes("latency") ? value.toFixed(0) : value.toFixed(3)}</span></td>; })}</tr>;
            })}</tbody>
          </table>
        </div>
      </section>

      <div className="evaluation-lower-grid">
        <section className="native-index-panel">
          <div className="section-heading"><div><span className="eyebrow">ISOLATED EXPERIMENT</span><h2>Native indexing paths</h2></div><span className="array-tag">DETERMINISTIC HASH / 128D</span></div>
          <div className="native-index-grid"><div className="native-head"><span>Framework</span><span>Chunks</span><span>Contain</span><span>Recall@1</span><span>Index ms</span></div>{nativeIndexing.map((row) => <div className="native-row" key={row.framework}><strong>{row.framework}</strong><span>{row.chunks.toFixed(2)}</span><span>{row.containment.toFixed(3)}</span><span>{row.recall.toFixed(3)}</span><span>{row.latency.toFixed(2)}</span></div>)}</div>
        </section>
        <aside className="interpretation-panel">
          <span className="eyebrow">READ BEFORE COMPARING</span>
          <h2>Measurements, not winners.</h2>
          <ol><li><span>01</span><p>Shared retrieval scores match by design; orchestration and generation behavior are the variables.</p></li><li><span>02</span><p>The corpus is intentionally small and synthetic. It validates the harness, not clinical performance.</p></li><li><span>03</span><p>Single-run latency includes local model warm state and cannot establish framework superiority.</p></li></ol>
        </aside>
      </div>
    </div>
  );
}
