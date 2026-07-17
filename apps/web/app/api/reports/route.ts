import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { parseComparisonReport } from "@/lib/reports";

export const dynamic = "force-dynamic";

const REPORT_FILE = "custom-vs-langchain-vs-langgraph-vs-llamaindex-vs-haystack-llama3.2-v1.md";

export async function GET() {
  const configuredRoot = process.env.RAGLAB_REPO_ROOT;
  const repositoryRoot = configuredRoot
    ? path.resolve(configuredRoot)
    : path.resolve(process.cwd(), "../..");
  const reportPath = path.join(
    /* turbopackIgnore: true */ repositoryRoot,
    "reports",
    "baselines",
    REPORT_FILE,
  );
  try {
    const markdown = await readFile(reportPath, "utf8");
    return NextResponse.json(
      parseComparisonReport(markdown, `reports/baselines/${REPORT_FILE}`),
    );
  } catch (error) {
    const detail = error instanceof Error ? error.name : "ReadError";
    return NextResponse.json(
      { error: { type: detail, message: "The committed comparison baseline is unavailable." } },
      { status: 503 },
    );
  }
}
