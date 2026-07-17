import { describe, expect, it } from "vitest";

import { drainSseBuffer, parseSseBlock } from "@/lib/sse";

describe("SSE parsing", () => {
  it("keeps an incomplete event and emits complete lifecycle events", () => {
    const drained = drainSseBuffer(
      'event: query.accepted\ndata: {"framework":"custom"}\n\n' +
        'event: query.result\ndata: {"answer":"grounded',
    );

    expect(drained.events).toEqual([
      { event: "query.accepted", data: { framework: "custom" } },
    ]);
    expect(drained.remainder).toContain("query.result");
  });

  it("rejects malformed JSON without exposing partial data", () => {
    expect(parseSseBlock("event: query.error\ndata: not-json")).toBeNull();
  });
});
