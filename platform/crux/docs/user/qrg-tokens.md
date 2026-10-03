# QRG — Token Optimization

Compression is automatic after Enterprise Ontology authorized the payload. There is no user switch.

## What it is

**Crux — Token Optimization (Headroom)** calls `POST /v1/compress` on the unpublished worker. Agents never receive the Headroom URL.

Bypass is trusted `CallKind` in `crux/compress.py` (extraction, embedding, ontology-admin). It is not:

- a request header
- `x-headroom-bypass`
- a line in the prompt
- a second “compress-off” HTTP call

## What you do not configure

- Do not publish Headroom to the host
- Do not point `chat_base_url` at Headroom
- Do not treat Headroom as memory, search, or a knowledge store

CCR markers fail-open to the original authorized payload.

## Measure (hypothesis)

```bat
set CRUX_AGENT_ID=agent-prod
set CRUX_CREDENTIAL_ID=cred-prod
python -m crux measure ontology
```

Cells A and C are `tokens_before` from the same authorized B and D asks. They are not a caller bypass. A zero `tokens_saved` on a short corpus is not a failure of the product — it is a measurement, not a guarantee.

Graph-first only when `find_entities` returns an id. Otherwise document fallback.
