# ADR-0005: User-facing module names

**Date**: 2026-09-07
**Status**: accepted
**Deciders**: product owner, session

## Context

Crux is already the product name (ADR-0001). Operators need module names that say what each piece does without implying a third access-control engine.

## Decision

User-facing names:

- **Crux — Enterprise Ontology (Utopia)** for the human console (`127.0.0.1:1516`)
- **Crux — Token Optimization (Headroom)** for the unpublished compressor

Compose services, Python classes, MCP paths, and env vars keep technical identifiers (`utopia`, `headroom`). Historical ADRs 0001–0004 are not rewritten.

## Consequences

The Ontology UI overlay changes title and banner only. In-app nav strings compiled into the pinned image may still say Utopia.
