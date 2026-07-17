# ADR 0006: Use scoped keyset pagination for growing resources

- Status: Accepted
- Date: 2026-07-17
- Deciders: RAGLab maintainers

## Context

Collection and document endpoints returned unbounded arrays, and ingestion jobs could only be fetched individually. These resources grow over time and need predictable response sizes without offset scans becoming slower or shifting under concurrent writes.

The initial assumptions are forward navigation, page sizes no larger than 100, immutable ordering fields, moderate PostgreSQL volume, and no requirement for a transactionally frozen snapshot across many requests. Pagination must not become an authorization substitute; authentication remains separate.

## Decision

Return `{items, next_cursor}` from collection, document, and collection-scoped ingestion-job list endpoints. Default to 20 items and validate a range of 1–100 at the HTTP boundary.

Order collections and jobs by `(created_at, id)` and documents by `(uploaded_at, id)`. Query one extra row to determine whether another page exists. Encode the last returned key as versioned URL-safe base64 JSON containing resource kind, collection scope, timestamp, and UUID. Validate the complete payload and reject malformed, cross-resource, or cross-collection reuse with HTTP 422.

Cursors are intentionally opaque but not signed or encrypted. They contain no secret, and changing a valid position cannot grant access that the endpoint itself would deny. Future authentication and tenancy checks must run independently of pagination.

## Options considered

### Unbounded arrays

Simple but unsafe as resources grow. Rejected.

### Offset and limit

Easy to expose, but deep offsets require increasing database work and concurrent inserts can shift later pages, causing duplicates or omissions.

### Signed or encrypted cursors

Useful when cursors contain sensitive state or enforce authorization. Current payloads contain only public ordering keys, so secret rotation and key management would add complexity without a security boundary.

### Scoped keyset cursors

Selected for bounded query work, deterministic timestamp ties, and explicit misuse detection.

## Consequences

- Page query cost stays bounded by the page limit and indexed ordering path rather than offset depth.
- API list responses change from arrays to page envelopes.
- Navigation is forward-only; arbitrary page numbers are not supported.
- UUID tie-breakers prevent gaps when timestamps are equal.
- Concurrent changes are visible according to key ordering rather than a frozen snapshot. A newly committed item whose key sorts before the current cursor will not appear in that traversal.
- Base64 makes cursors transport-safe, not confidential or tamper-proof. Authorization must never trust cursor contents.
- At larger scale, revisit composite indexes based on query plans, signed cursors if private filters enter the payload, bidirectional navigation, consistent snapshot exports, and database-generated monotonic ordering keys.
