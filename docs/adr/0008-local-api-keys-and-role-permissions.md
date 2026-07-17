# ADR 0008: Local API keys and role permissions

## Status

Accepted

## Context

RAGLab now exposes ingestion, querying, and destructive document deletion through one API. It needs an authentication and authorization boundary without adding a paid identity service, account dependency, or remote call to the local-first runtime. The backend has no user or tenant data model yet, and the future browser application will likely need a different session-oriented identity flow.

The current assumptions are a local demo or small trusted team, a small number of service credentials, deployment-wide access to the same corpus, environment-based secret delivery, coordinated API process restarts for rotation, and TLS termination outside the application in non-local environments. Health probes must remain available without credentials.

## Decision

Add optional bearer API-key authentication backed by environment configuration and coarse role-based permissions.

- Development and test environments may explicitly run with authentication disabled. Disabled mode resolves requests to a named local admin principal so authorization checks still execute.
- Staging and production fail configuration validation unless authentication is enabled and at least one key is present.
- Each configured credential has a non-secret subject name, a `viewer`, `editor`, or `admin` role, and a secret of 32 to 256 characters.
- The authenticator immediately hashes keys with SHA-256 and retains only digests internally. Authentication compares every configured digest using constant-time comparison; settings models redact secret values from representations.
- `viewer` can inspect shared metadata and query; `editor` can additionally create collections and ingest; `admin` can additionally delete documents.
- `/health/live`, `/health/ready`, and OpenAPI discovery remain public. Every application-data endpoint enforces a declared permission. `/auth/me` exposes the resolved subject, role, and permissions but never the key.
- Authentication failures use HTTP 401 and a Bearer challenge. Authorization failures use HTTP 403 and the shared safe error envelope.

```text
Bearer key ── SHA-256 + constant-time match ── named principal + role
                                                     │
                                                     ▼
                                      declared route permission ── handler

Public health probes ────────────────────────────────────────────── handler
```

## Alternatives considered

### Local username/password accounts with JWT access tokens

This is familiar for browser applications but requires password recovery, account lifecycle, token signing and rotation, refresh-token policy, and additional persistence before the project has a user model. It is deferred until the frontend and tenancy requirements are concrete.

### External OAuth or hosted identity provider

This provides mature login and federation but introduces an account, network dependency, configuration burden, and potentially paid service into the default path. A standards-based provider remains the preferred direction for a public multi-user deployment.

### One unscoped static token

This is simpler but cannot distinguish read, write, and destructive operations. Named keys plus role-derived permissions add useful authorization without a database migration.

### Persist API keys in PostgreSQL

Database-backed creation, revocation, expiry, and auditing would improve operations, but also requires a secure bootstrap path and management API. Environment configuration is sufficient for the current bounded deployment assumption.

## Consequences

- The authentication path is local, deterministic, and free to operate.
- OpenAPI documents the Bearer scheme and protected route dependencies.
- Staging and production cannot accidentally start with an open API.
- A leaked key remains valid until configuration changes and API processes restart.
- Roles grant deployment-wide access; there is no collection ownership or tenant isolation.
- Key names are identity labels, not human accounts, and authentication events are not yet audit records.

Revisit this decision when RAGLab adds a browser login, multiple untrusted users, collection ownership, independent revocation, expiring credentials, audit requirements, or public internet exposure. At that point prefer an OpenID Connect provider with short-lived tokens, issuer and audience validation, per-collection policy checks, rate limits, security event logging, and a migration path that preserves the current permission vocabulary.
