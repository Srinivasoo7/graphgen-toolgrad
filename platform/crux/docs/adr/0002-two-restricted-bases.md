# ADR-0002: Two restricted Utopia bases

**Date**: 2026-09-06
**Status**: accepted
**Deciders**: product owner, session

## Context

Prod vs staging inside one knowledge base needs object-level enforcement on search, graph, documents, citations, counts, and derived facts. That is not proven.

## Decision

Release A uses two restricted Utopia bases (Production, Staging) sharing ontology packs. Session may only select a subset of already-granted bases. Same-base filters (e.g. one incident) wait until the leak acceptance test passes on live Utopia.

## Alternatives Considered

### One base with object tags
- **Why not**: Leak surfaces (counts, derived facts) are easy to get wrong.

### Ontology per agent
- **Why not**: Duplicated terms and conflicting definitions.

## Consequences

### Positive
- Isolation matches Utopia’s real ACL unit (the knowledge base).

### Negative
- Shared entities across prod/staging need two writes or a later filter.

### Risks
- Operators create an open base by mistake. Mitigation: bootstrap checklist.
