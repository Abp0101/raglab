import { describe, expect, it } from "vitest";

import { parsePrometheus, sumMetric } from "@/lib/metrics";

describe("Prometheus parsing", () => {
  it("parses bounded labels and sums one metric family", () => {
    const samples = parsePrometheus(`
# HELP raglab_http_requests_total Completed requests.
raglab_http_requests_total{method="GET",route="/health/live",status_class="2xx"} 4
raglab_http_requests_total{method="POST",route="/query",status_class="5xx"} 1
raglab_dependency_up{dependency="qdrant"} 0
`);

    expect(sumMetric(samples, "raglab_http_requests_total")).toBe(5);
    expect(samples.at(-1)).toEqual({
      name: "raglab_dependency_up",
      labels: { dependency: "qdrant" },
      value: 0,
    });
  });
});
