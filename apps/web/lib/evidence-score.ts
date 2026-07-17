import type { RetrievedChunk } from "@/types/api";

export interface EvidenceSignal {
  digits: number;
  label: string;
  value: number | null;
}

export function evidenceSignalFor(chunk: RetrievedChunk | undefined): EvidenceSignal {
  if (!chunk) return { label: "Selected evidence score", value: null, digits: 3 };
  if (chunk.reranker_score != null) {
    return { label: "Selected reranker raw score", value: chunk.reranker_score, digits: 3 };
  }
  if (chunk.fusion_score != null) {
    return { label: "Selected fusion rank score", value: chunk.fusion_score, digits: 4 };
  }
  if (chunk.dense_score != null) {
    return { label: "Selected dense similarity", value: chunk.dense_score, digits: 3 };
  }
  if (chunk.sparse_score != null) {
    return { label: "Selected BM25 score", value: chunk.sparse_score, digits: 3 };
  }
  return { label: "Selected retrieval score", value: chunk.relevance_score, digits: 3 };
}

export function relativeScorePercent(target: RetrievedChunk, chunks: RetrievedChunk[]): number {
  const value = evidenceSignalFor(target).value;
  const scores = chunks
    .map((chunk) => evidenceSignalFor(chunk).value)
    .filter((score): score is number => score != null && Number.isFinite(score));

  if (value == null || !Number.isFinite(value) || scores.length === 0) return 8;
  const minimum = Math.min(...scores);
  const maximum = Math.max(...scores);
  if (minimum === maximum) return 100;
  return 8 + ((value - minimum) / (maximum - minimum)) * 92;
}
