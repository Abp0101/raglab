export type Framework =
  | "custom"
  | "langchain"
  | "langgraph"
  | "llamaindex"
  | "haystack";

export type RetrievalMode = "dense" | "sparse" | "hybrid";
export type EvidenceStatus = "sufficient" | "insufficient" | "conflicting";

export interface CursorPage<T> {
  items: T[];
  next_cursor: string | null;
}

export interface Collection {
  collection_id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  document_count: number;
}

export interface DocumentRecord {
  document_id: string;
  collection_id: string;
  file_name: string;
  display_title: string;
  authors: string[];
  source_url: string | null;
  uploaded_at: string;
  publication_date: string | null;
  file_type: string;
  content_hash: string;
  page_count: number | null;
  status: "pending" | "processing" | "ready" | "failed" | "deleting";
}

export interface IngestionJob {
  job_id: string;
  collection_id: string;
  file_name: string;
  status: "queued" | "processing" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  attempt_count: number;
  lease_expires_at: string | null;
  error: { type: string; message: string } | null;
}

export interface PipelineSummary {
  framework: Framework;
  available: boolean;
  capabilities: {
    ingestion: boolean;
    dense_retrieval: boolean;
    sparse_retrieval: boolean;
    hybrid_retrieval: boolean;
    reranking: boolean;
    metadata_filtering: boolean;
    streaming: boolean;
    agentic: boolean;
  };
  config: {
    retrieval_mode: RetrievalMode;
    top_k: number;
    candidate_k: number;
    rerank: boolean;
    rerank_top_k: number;
    evidence_threshold: number;
    max_context_tokens: number;
  };
}

export interface ChunkMetadata {
  document_id: string;
  collection_id: string;
  file_name: string;
  display_title: string;
  authors: string[];
  source_url: string | null;
  uploaded_at: string;
  publication_date: string | null;
  file_type: string;
  page_number: number | null;
  section_heading: string | null;
  chunk_index: number;
  parent_chunk_id: string | null;
  content_hash: string;
}

export interface RetrievedChunk {
  chunk: {
    chunk_id: string;
    text: string;
    metadata: ChunkMetadata;
    token_count: number | null;
  };
  rank: number;
  relevance_score: number | null;
  dense_score: number | null;
  sparse_score: number | null;
  fusion_score: number | null;
  reranker_score: number | null;
}

export interface Citation {
  document_id: string;
  document_title: string;
  page_number: number | null;
  section_heading: string | null;
  chunk_id: string;
  quoted_text: string;
  relevance_score: number | null;
  reranker_score: number | null;
}

export interface RAGResponse {
  answer: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  framework: Framework;
  model: string;
  latency: {
    total_ms: number;
    retrieval_ms: number;
    reranking_ms: number;
    generation_ms: number;
  };
  usage: {
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
    estimated_cost_usd: number | null;
    llm_calls: number;
    retrieval_iterations: number;
  };
  evidence_status: EvidenceStatus;
  confidence: number | null;
  warnings: string[];
}

export interface QueryPayload {
  query: string;
  framework: Framework;
  collection_id: string;
  top_k: number;
  retrieval_mode: RetrievalMode;
  rerank: boolean;
  temperature: number;
  debug: boolean;
}

export interface HealthReadiness {
  status: "ok" | "degraded";
  dependencies: Record<string, boolean>;
}

export interface ApiErrorEnvelope {
  error?: { type?: string; message?: string };
}

export interface BenchmarkMetric {
  name: string;
  values: Record<string, number>;
}

export interface BenchmarkReport {
  title: string;
  runDate: string;
  model: string;
  cost: string;
  frameworks: string[];
  metrics: BenchmarkMetric[];
  source: string;
}
