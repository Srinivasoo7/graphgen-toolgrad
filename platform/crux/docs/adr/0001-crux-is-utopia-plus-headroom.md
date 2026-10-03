# ADR-0001: Crux is Utopia plus Headroom

**Date**: 2026-09-06
**Status**: accepted
**Deciders**: product owner, session

## Context

Combining headroomlabs-ai/headroom and deeplethe/utopia. Early drafts named a third “Crux gateway/filter” that would re-implement access control.

## Decision

Crux is the **product name**. Utopia owns ontology, graph, review, and access. Headroom compresses payloads Utopia already authorized. Packaging (compose, enrollment records, instrumented MCP log) lives in this repo.

## Alternatives Considered

### Merge the two codebases
- **Why not**: Different runtimes and release cadences; permanent rebase tax.

### Crux as an authorization proxy in front of Utopia
- **Why not**: Duplicates Utopia roles/tokens; agents with a side-door PAT bypass it; naming implied Crux decides knowledge access.

## Consequences

### Positive
- One policy engine (Utopia). Clear ownership.

### Negative
- Product glue must not grow into a second ACL.

### Risks
- Side-door Utopia MCP with a wider PAT. Mitigation: loopback bind, no DB/Headroom creds in agent configs.
