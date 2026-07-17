import type { BenchmarkReport } from "@/types/api";

export function parseComparisonReport(markdown: string, source: string): BenchmarkReport {
  const title = markdown.match(/^#\s+(.+)$/m)?.[1] ?? "Framework comparison";
  const runDate = markdown.match(/^- Run date:\s+(.+)$/m)?.[1] ?? "Unknown";
  const model = markdown.match(/^- Model:\s+(.+)$/m)?.[1]?.replaceAll("`", "") ?? "Local";
  const cost = markdown.match(/^- Paid API cost:\s+(.+)$/m)?.[1]?.replaceAll("`", "") ?? "$0.00";
  const lines = markdown.split("\n");
  const tableStart = lines.findIndex((line) => line.startsWith("| Metric |"));
  if (tableStart < 0) throw new Error("comparison report metric table is missing");
  const frameworks = lines[tableStart]
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean)
    .slice(1);
  const metricLines: string[] = [];
  for (const line of lines.slice(tableStart + 2)) {
    if (!line.startsWith("|")) break;
    metricLines.push(line);
  }
  const metrics = metricLines
    .map((line) => line.split("|").map((cell) => cell.trim()).filter(Boolean))
    .filter((cells) => cells.length === frameworks.length + 1)
    .map(([name, ...cells]) => ({
      name,
      values: Object.fromEntries(frameworks.map((framework, index) => [framework, Number(cells[index])])),
    }));
  return { title, runDate, model, cost, frameworks, metrics, source };
}
