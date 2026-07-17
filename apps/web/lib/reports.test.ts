import { describe, expect, it } from "vitest";

import { parseComparisonReport } from "@/lib/reports";

describe("committed comparison report parser", () => {
  it("preserves framework columns and exact observed values", () => {
    const report = parseComparisonReport(
      `# Local comparison

- Run date: 2026-07-17
- Model: local \`llama3.2:latest\`
- Paid API cost: \`$0.00\`

| Metric | Custom | LangChain |
| --- | ---: | ---: |
| Citation recall | 1.0000 | 0.8333 |
| Mean latency (ms) | 2726.7 | 1111.8 |

Interpretation follows.
`,
      "reports/baselines/example.md",
    );

    expect(report.frameworks).toEqual(["Custom", "LangChain"]);
    expect(report.metrics[0].values).toEqual({ Custom: 1, LangChain: 0.8333 });
    expect(report.cost).toBe("$0.00");
  });
});
