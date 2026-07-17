import { describe, expect, it } from "vitest";

import { evidenceSignalFor, relativeScorePercent } from "@/lib/evidence-score";
import type { RetrievedChunk } from "@/types/api";

function result(rerankerScore: number, rank: number): RetrievedChunk {
  return {
    chunk: {
      chunk_id: `chunk-${rank}`,
      text: "Evidence",
      metadata: {
        document_id: "document",
        collection_id: "collection",
        file_name: "test.pdf",
        display_title: "Test",
        authors: [],
        source_url: null,
        uploaded_at: "2026-07-17T00:00:00Z",
        publication_date: null,
        file_type: "application/pdf",
        page_number: 1,
        section_heading: "Test",
        chunk_index: rank - 1,
        parent_chunk_id: null,
        content_hash: "hash",
      },
      token_count: 1,
    },
    rank,
    relevance_score: rerankerScore,
    dense_score: 0.5,
    sparse_score: 2,
    fusion_score: 0.03,
    reranker_score: rerankerScore,
  };
}

describe("evidence score presentation", () => {
  it("keeps unbounded reranker logits as raw values", () => {
    expect(evidenceSignalFor(result(6.768, 1))).toEqual({
      label: "Selected reranker raw score",
      value: 6.768,
      digits: 3,
    });
  });

  it("normalizes positive and negative raw scores only for the visual gauge", () => {
    const high = result(6.768, 1);
    const middle = result(-1.5, 2);
    const low = result(-8.2, 3);
    expect(relativeScorePercent(high, [high, middle, low])).toBe(100);
    expect(relativeScorePercent(low, [high, middle, low])).toBe(8);
    expect(relativeScorePercent(middle, [high, middle, low])).toBeGreaterThan(8);
    expect(relativeScorePercent(middle, [high, middle, low])).toBeLessThan(100);
  });
});
