# ADR 0005: Coordinate ingestion workers with PostgreSQL leases

- Status: Accepted
- Date: 2026-07-17
- Deciders: RAGLab maintainers

## Context

The original background manager scheduled process-local tasks and requeued every `processing` job at startup. With two Uvicorn processes, the second process could reset and execute work still owned by the first. RAGLab needs crash recovery and horizontal API workers without introducing a paid queue or another required service.

Assumptions for this milestone are modest ingestion volume, PostgreSQL already being mandatory, PDFs taking seconds to minutes to process, and at-least-once execution being acceptable when storage operations remain idempotent or compensating. The design must remain fully local and free.

## Decision

Store `lease_owner`, `lease_expires_at`, and `attempt_count` on each durable ingestion job. Every API process creates a unique owner ID and a configured number of pollers. A poller atomically selects the oldest queued job—or a processing job with an expired lease—using `FOR UPDATE SKIP LOCKED`, changes it to `processing`, and assigns itself a lease.

Use PostgreSQL's clock for acquisition, renewal, expiry, and terminal-write checks so host clock skew cannot cause an early takeover. Renew active work every third of the lease window. A completion or failure is accepted only when the job is still processing, the owner matches, and the lease remains live. Graceful cancellation releases the job to `queued`; process death requires no cleanup because another worker can reclaim it after expiry.

Retain source bytes until terminal completion or failure. Clear bytes and lease fields together on a successful terminal transition. Expose attempt count and lease expiry in the public job model, but never expose the owner identifier.

## Options considered

### Process-local task ownership

Simple, but unsafe once more than one API process starts. Rejected.

### Redis queue

Redis is already present, but reliable visibility timeouts, acknowledgements, retries, and persistence would either duplicate job state or require adopting and operating a queue library. It adds complexity without a demonstrated throughput need.

### PostgreSQL advisory locks

They provide ownership while a connection remains open, but holding a database connection throughout PDF parsing, embeddings, and indexing is wasteful and crash visibility is less explicit to API clients.

### PostgreSQL rows with expiring leases

Selected because queue state, observability, recovery, and ownership checks remain transactional in the existing source of truth.

## Consequences

- Multiple API processes can safely compete for queued work.
- Worker crashes recover automatically after a bounded lease delay.
- The system guarantees at-least-once execution. A crash between external-store writes and the terminal PostgreSQL update can repeat work.
- Polling adds a small steady PostgreSQL load bounded by process count, concurrency, and `RAGLAB_INGESTION_POLL_SECONDS`.
- A claimable index supports status/expiry scans; `SKIP LOCKED` avoids convoying between workers.
- A lease that is too short increases false takeovers during event-loop or database stalls; one that is too long slows crash recovery. The default is 60 seconds with 20-second renewal.
- At higher sustained ingestion volume, revisit a dedicated worker deployment, queue depth metrics, retry limits/dead-letter state, per-tenant fairness, and possibly a purpose-built broker. The owner-checked job contract can remain the API-facing source of truth.
