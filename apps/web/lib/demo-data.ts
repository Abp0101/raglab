import type {
  Collection,
  DocumentRecord,
  IngestionJob,
  PipelineSummary,
  RAGResponse,
} from "@/types/api";

export const demoCollections: Collection[] = [
  {
    collection_id: "2f7aa253-3581-5c59-b92a-9b0cc520bbca",
    name: "Wearable rehabilitation evidence",
    description: "IMU, plantar force, and clinical validation studies.",
    created_at: "2026-07-14T09:00:00Z",
    updated_at: "2026-07-17T15:12:00Z",
    document_count: 12,
  },
  {
    collection_id: "703510bb-8afb-5d88-a7b3-775300c548c4",
    name: "Clinical AI governance",
    description: "Drift monitoring, retrospective validation, and deployment guidance.",
    created_at: "2026-07-15T11:20:00Z",
    updated_at: "2026-07-17T14:36:00Z",
    document_count: 8,
  },
];

export const demoDocuments: DocumentRecord[] = [
  {
    document_id: "59497828-cfc9-51ed-8b0d-16f31f4e89a4",
    collection_id: demoCollections[0].collection_id,
    file_name: "imu-gait-validation.pdf",
    display_title: "Validation of wearable IMU gait measurements",
    authors: ["A. Rahman", "L. Chen"],
    source_url: null,
    uploaded_at: "2026-07-17T12:05:00Z",
    publication_date: "2025-11-03",
    file_type: "application/pdf",
    content_hash: "a".repeat(64),
    page_count: 14,
    status: "ready",
  },
  {
    document_id: "778c0ee0-5811-57c7-a8a7-e8fda693010a",
    collection_id: demoCollections[0].collection_id,
    file_name: "plantar-force-calibration.pdf",
    display_title: "Plantar force sensor calibration protocol",
    authors: ["M. Okafor"],
    source_url: null,
    uploaded_at: "2026-07-17T11:40:00Z",
    publication_date: "2024-08-12",
    file_type: "application/pdf",
    content_hash: "b".repeat(64),
    page_count: 9,
    status: "ready",
  },
  {
    document_id: "c45f936c-ceab-5352-881e-22d3204c1653",
    collection_id: demoCollections[0].collection_id,
    file_name: "rehab-device-safety.pdf",
    display_title: "Safety controls for home rehabilitation devices",
    authors: ["S. Patel", "J. Moore"],
    source_url: null,
    uploaded_at: "2026-07-17T15:12:00Z",
    publication_date: null,
    file_type: "application/pdf",
    content_hash: "c".repeat(64),
    page_count: 22,
    status: "processing",
  },
];

export const demoJobs: IngestionJob[] = [
  {
    job_id: "9f0194e5-bfa6-5281-9705-888cd6605397",
    collection_id: demoCollections[0].collection_id,
    file_name: "rehab-device-safety.pdf",
    status: "processing",
    created_at: "2026-07-17T15:12:00Z",
    updated_at: "2026-07-17T15:12:16Z",
    attempt_count: 1,
    lease_expires_at: "2026-07-17T15:13:16Z",
    error: null,
  },
];

const capabilityBase = {
  ingestion: true,
  dense_retrieval: true,
  sparse_retrieval: true,
  hybrid_retrieval: true,
  reranking: true,
  metadata_filtering: true,
  streaming: true,
  agentic: false,
};

const configBase = {
  retrieval_mode: "hybrid" as const,
  top_k: 5,
  candidate_k: 20,
  rerank: true,
  rerank_top_k: 5,
  evidence_threshold: 0.5,
  max_context_tokens: 6000,
};

export const demoPipelines: PipelineSummary[] = [
  "custom",
  "langchain",
  "langgraph",
  "llamaindex",
  "haystack",
].map((framework) => ({
  framework: framework as PipelineSummary["framework"],
  available: true,
  capabilities: { ...capabilityBase, agentic: framework === "langgraph" },
  config: configBase,
}));

const firstDocument = demoDocuments[0];

export const demoResponse: RAGResponse = {
  answer:
    "The wearable system sampled the lower-limb IMUs at 100 Hz and synchronized them to the plantar-force acquisition clock before segmentation. The validation protocol reports this timing configuration for the reference gait trials.",
  citations: [
    {
      document_id: firstDocument.document_id,
      document_title: firstDocument.display_title,
      page_number: 6,
      section_heading: "Acquisition protocol",
      chunk_id: "86e863d8-cff6-522c-9dd9-89647520619b",
      quoted_text:
        "All six inertial measurement units were sampled at 100 Hz and synchronized to the plantar-force acquisition clock before gait-cycle segmentation.",
      relevance_score: 0.93,
      reranker_score: 0.88,
    },
  ],
  retrieved_chunks: [
    {
      chunk: {
        chunk_id: "86e863d8-cff6-522c-9dd9-89647520619b",
        text: "All six inertial measurement units were sampled at 100 Hz and synchronized to the plantar-force acquisition clock before gait-cycle segmentation. Drift correction was applied once per recording session.",
        metadata: {
          document_id: firstDocument.document_id,
          collection_id: firstDocument.collection_id,
          file_name: firstDocument.file_name,
          display_title: firstDocument.display_title,
          authors: firstDocument.authors,
          source_url: null,
          uploaded_at: firstDocument.uploaded_at,
          publication_date: firstDocument.publication_date,
          file_type: firstDocument.file_type,
          page_number: 6,
          section_heading: "Acquisition protocol",
          chunk_index: 18,
          parent_chunk_id: null,
          content_hash: firstDocument.content_hash,
        },
        token_count: 42,
      },
      rank: 1,
      relevance_score: 0.93,
      dense_score: 0.78,
      sparse_score: 4.17,
      fusion_score: 0.0325,
      reranker_score: 0.88,
    },
    {
      chunk: {
        chunk_id: "f02ca6cd-5037-5997-b175-8d06db32fd32",
        text: "The plantar-force array was calibrated at five known loads. A held-out load was used to quantify calibration error before participant testing.",
        metadata: {
          document_id: demoDocuments[1].document_id,
          collection_id: demoDocuments[1].collection_id,
          file_name: demoDocuments[1].file_name,
          display_title: demoDocuments[1].display_title,
          authors: demoDocuments[1].authors,
          source_url: null,
          uploaded_at: demoDocuments[1].uploaded_at,
          publication_date: demoDocuments[1].publication_date,
          file_type: demoDocuments[1].file_type,
          page_number: 3,
          section_heading: "Calibration",
          chunk_index: 7,
          parent_chunk_id: null,
          content_hash: demoDocuments[1].content_hash,
        },
        token_count: 29,
      },
      rank: 2,
      relevance_score: 0.61,
      dense_score: 0.65,
      sparse_score: 1.22,
      fusion_score: 0.0298,
      reranker_score: 0.34,
    },
  ],
  framework: "langgraph",
  model: "llama3.2:latest",
  latency: { total_ms: 1332, retrieval_ms: 44, reranking_ms: 118, generation_ms: 1170 },
  usage: {
    prompt_tokens: 1068,
    completion_tokens: 58,
    total_tokens: 1126,
    estimated_cost_usd: 0,
    llm_calls: 1,
    retrieval_iterations: 1,
  },
  evidence_status: "sufficient",
  confidence: 0.91,
  warnings: [],
};

export const demoMetrics = `
raglab_http_requests_total{method="GET",route="/collections",status_class="2xx"} 48
raglab_http_requests_total{method="POST",route="/query",status_class="2xx"} 21
raglab_http_requests_total{method="GET",route="/health/ready",status_class="2xx"} 96
raglab_http_requests_total{method="POST",route="/query",status_class="5xx"} 1
raglab_errors_total{operation="/query",error_type="ProviderUnavailable"} 1
raglab_ingestion_jobs_total{outcome="completed"} 12
raglab_ingestion_jobs_total{outcome="claimed"} 13
raglab_dependency_up{dependency="postgres"} 1
raglab_dependency_up{dependency="qdrant"} 1
raglab_dependency_up{dependency="redis"} 1
`;
