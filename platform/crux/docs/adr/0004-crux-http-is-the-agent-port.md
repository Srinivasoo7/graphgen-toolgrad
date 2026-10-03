# ADR-0004: Crux HTTP is the agent port

**Date**: 2026-09-06
**Status**: accepted
**Deciders**: product owner, session

## Context

Agents must not receive a raw Utopia URL plus a wider PAT, database URL, or Headroom credentials. Admin work stays in the Utopia UI.

## Decision

Publish Crux on loopback (`127.0.0.1:8788`). Agents call `POST /v1/ask`. Crux authorizes via Utopia-shaped enrollment, calls Utopia MCP with the stored PAT, then Headroom `/v1/compress`. Headroom stays unpublished. Utopia UI stays on `127.0.0.1:1516` for humans.

## Alternatives Considered

### Publish Headroom on loopback for the host CLI
- **Why not**: Gives a compress side door on the host. The Crux container already reaches Headroom on the compose network.

### Agents call Utopia MCP directly
- **Why not**: Side door. Instrumented audit and compression would be optional.

## Consequences

### Positive
- One agent-facing port. Postgres and Headroom stay off the agent network.

### Negative
- Crux stores the Utopia PAT so the agent never holds it.

### Risks
- A stolen Crux admin token can enroll new agents. Mitigation: admin token is compose-local and not given to agents.
