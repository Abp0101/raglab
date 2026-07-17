# ADR 0007: Retryable cross-store document deletion

## Status

Accepted

## Context

RAGLab stores authoritative document and chunk records in PostgreSQL, dense retrieval points in Qdrant, and lexical retrieval entries in Redis. These systems do not share a transaction. Deletion must not report success while retrievable content remains, and a partial provider outage must not discard the chunk identifiers and metadata required to retry cleanup.

The current assumptions are moderate document sizes, bounded chunk counts, operator- or single-user initiated deletion, idempotent Qdrant and Redis delete operations, and no legal-hold, retention-policy, tenant-authorization, or bulk-deletion requirement. Ingestion and deletion can arrive concurrently through multiple API processes.

## Decision

Use PostgreSQL as the deletion coordinator and durable source of retry data.

1. Lock the document row and transition a `ready` or `failed` document to `deleting`. A document already in `deleting` resumes the same workflow. Reject `pending` and `processing` documents with a conflict.
2. Load all document chunks from PostgreSQL.
3. Delete their point IDs from Qdrant.
4. Delete their lexical entries from Redis.
5. Delete the PostgreSQL document last; relational cascade removes its chunks.

The coordinator returns success only after all five steps complete. Qdrant and Redis deletion are idempotent, so a retry can repeat completed external steps. If any storage operation fails, the API returns a safe provider-unavailable error and PostgreSQL remains in `deleting` state unless its final delete already committed. This is a retryable saga with ordered compensation semantics, not a distributed transaction.

## Alternatives considered

### Delete PostgreSQL first

This removes the only complete list of chunk IDs and metadata before external cleanup succeeds. Recovery would require broad provider scans or leaked index entries, so this ordering was rejected.

### Run all deletes concurrently

This reduces latency but makes failure ordering nondeterministic and can remove PostgreSQL while an external store fails. The small latency saving does not justify weaker recovery semantics.

### Introduce a message broker and deletion worker now

A durable outbox and worker would improve automatic retry and observability, but adds operational machinery beyond current volume. The synchronous coordinator establishes the state machine and idempotent adapter contract that a future worker can reuse.

### Use a distributed transaction

PostgreSQL, Qdrant, and Redis do not expose a practical shared two-phase commit boundary. Emulating one would add complexity without eliminating provider-specific failure modes.

## Consequences

- Successful deletion means the document has been removed from all three stores.
- A failed request is safe to retry while the PostgreSQL record remains.
- Retrieval can briefly observe asymmetric indexes between external deletion steps.
- `deleting` documents remain visible through metadata reads until cleanup completes, making incomplete work diagnosable.
- Active ingestion must finish before deletion begins.
- The API response reports the relational chunk count, not independent provider counts.

At higher deletion volume or stricter compliance requirements, revisit a transactional outbox, dedicated deletion workers, exponential retry with dead-letter state, tombstone filtering in retrieval, audit records, bulk operations, retention and legal-hold policy, tenant authorization, and metrics for documents stuck in `deleting`.
