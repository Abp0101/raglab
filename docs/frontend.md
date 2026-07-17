# Evidence workbench

The RAGLab web client is a Next.js App Router application in `apps/web`. It is an inspection surface for the existing local API, not a second data or inference plane. Its visual language is a research instrument: ruled paper, explicit channels, machine-readable labels, sharp panel boundaries, and evidence provenance kept beside each answer.

## Surfaces

| Route | Purpose | Local API inputs |
| --- | --- | --- |
| `/` | Corpus, pipeline, cost, and reference-trace overview | collections, pipelines, health |
| `/query` | Run a shared query and inspect the streamed answer, citations, ranked chunks, scores, and latency | collections, pipelines, `/query/stream` |
| `/library` | Create collections, upload PDFs, inspect document state, and follow ingestion jobs | collections, documents, ingestion jobs |
| `/evaluation` | Read the committed five-framework baseline and native-indexing observations | committed baseline report |
| `/operations` | Inspect bounded readiness and Prometheus process metrics with a safe runbook | `/health/ready`, `/metrics` |

The evaluation screen deliberately says “not a leaderboard.” Bars are normalized only within the selected measurement and every plotted value is also printed exactly. The operations screen uses route templates and sanitized error types; it does not display prompts, document contents, credentials, or unbounded metric labels.

## Local setup

Start the API and its local dependencies from the repository root, then install and run the workbench:

```bash
cp apps/web/.env.example apps/web/.env.local
make web-install
make web-dev
```

Open <http://localhost:3000>. The default API root is <http://localhost:8000>, matching the backend CORS default.

The connection button in the top bar can change the API root for the current browser. An optional bearer key is stored in `sessionStorage`, so it is scoped to the current tab and disappears when that session ends. Do not put API keys in a `NEXT_PUBLIC_` variable.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_RAGLAB_API_URL` | `http://localhost:8000` | Initial browser-visible local API root |
| `NEXT_PUBLIC_RAGLAB_DEMO_MODE` | `false` | Enables clearly labelled sample records when the API is unavailable |
| `NEXT_TELEMETRY_DISABLED` | `1` | Disables Next.js telemetry; also fixed in npm lifecycle scripts |
| `RAGLAB_REPO_ROOT` | inferred from `apps/web` | Server-only repository root used to read the committed baseline report |

Demo mode does not simulate successful mutations. Collection creation, PDF ingestion, and deletion remain blocked while preview fixtures are active, and every preview surface is visibly labelled.

## Data and streaming contract

Browser-side API types in `apps/web/types/api.ts` mirror the backend collection, document, job, pipeline, query, and evaluation shapes. `apps/web/lib/api.ts` centralizes safe fetch behavior and bearer-key handling. The query page reads Server-Sent Events incrementally: progress events update the run state and the terminal response replaces the reference trace only after a complete canonical response arrives.

The evaluation route parses the committed Markdown baseline on the server. For a standalone deployment, ship the `reports/baselines` directory with the application and point `RAGLAB_REPO_ROOT` at the repository-content root.

## Visual and accessibility rules

The interface uses locally installed Instrument Sans and IBM Plex Mono packages. It has no external font request, analytics client, stock illustration, component kit, or decorative AI imagery. The detailed token and component contract is in [`apps/web/DESIGN_SYSTEM.md`](../apps/web/DESIGN_SYSTEM.md).

Key interaction rules:

- landmarks, visible labels, heading order, tables, and native controls carry structure before ARIA is added;
- active framework and evidence states use text, geometry, and `aria-pressed`, not colour alone;
- keyboard focus is always visible, motion is removed under `prefers-reduced-motion`, and a skip link targets the workspace;
- the side rail becomes a five-key bottom instrument bar below 900 px;
- dense matrices scroll inside their own bounded region rather than forcing page-level horizontal scrolling.

## Verification

```bash
make web-check
```

This runs ESLint, strict TypeScript checking, Vitest component/parser tests, and a production Next.js build. CI runs the same gates with Node.js 22 and `npm ci`. The responsive acceptance pass covers a 1280 px desktop viewport and a 390 × 844 mobile viewport across all five routes, including the framework switch and connection dialog.

All application data and AI execution remain on the configured local RAGLab stack. The workbench adds no metered API call or hosted telemetry path.
