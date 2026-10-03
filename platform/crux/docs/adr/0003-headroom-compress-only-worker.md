# ADR-0003: Headroom is a compress-only worker

**Date**: 2026-09-06
**Status**: accepted
**Deciders**: product owner, session

## Context

Stock `headroom proxy` also forwards to the provider. Pointing Utopia `chat_base_url` at it would compress extraction and allow `x-headroom-bypass` from callers.

## Decision

Call Headroom `POST /v1/compress` only after Utopia authorized the payload. Release A: `--no-ccr`, Kompress off, no host publish. Bypass kinds (extraction, embedding, ontology-admin) are trusted `CallKind` config — never request headers or prompts.

## Alternatives Considered

### Utopia chat_base_url → Headroom proxy
- **Why not**: Extraction/admin share that URL; CCR/stream bugs; header bypass.

### Headroom memory as knowledge
- **Why not**: Ungoverned second world model.

## Consequences

### Positive
- Compression cannot see unauthorized content. Extraction stays on the provider.

### Negative
- `/v1/compress` is stateless; callers must pin `frozen_message_count`.

### Risks
- `HEADROOM_COMPRESS_ALLOW_REMOTE=1` on the compose network. Mitigation: no published port; require `HEADROOM_PROXY_TOKEN`.
