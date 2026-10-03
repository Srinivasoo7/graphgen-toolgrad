# Status

**Phase:** Release A complete (two restricted bases, leak surfaces, measure A–D).

## Healthy

- ECC Cursor adapter installed (`ecc-universal` 2.2.0).
- Utopia pin `33e12d97`: MCP `remember` is propose-only; write = token write ∩ editor.
- Tests: 106 passed. Live two-base isolation covers search, entities, docs, and facts.
- Docker stack up. Agent SDK, session subset, readiness, tool allowlist.
- `GET /v1/enrollments` (admin) lists bindings without PATs.
- `CruxClient.remember` is propose; read-only agents are denied.
- Ask envelope includes `text` (compressed reply) so agents need not parse JSON-RPC.
- Enroll refuses bases outside bootstrapped Production/Staging when `bootstrap.json` is present.
- `GET /v1/openapi.json` describes the agent port.
- `python -m crux ask` / `entities` / `facts` / `propose` / `audit` / `measure` / `readiness` for operator demo. Measure cells A/C are `tokens_before` from the same authorized B/D ask — not a caller bypass flag.
- Headroom CCR markers fail-open to the original authorized payload.
- Readiness `blockers` list missing chat model and Headroom chat_url (set models in Utopia UI).
- Chat/embed configured in Enterprise Ontology (not Headroom). Production and Staging graph extract are live. Cross-base leak surfaces stay empty for the other base.
- Operator console at `GET /` on `:8788`. QRGs in `docs/user/`. Pitch at `GET /pitch`.

## Blockers

- Same-base object filters still deferred (after leak acceptance; ADR-0002).

## Delete-zone (do not recreate)

- A third “Crux filter / gateway authz” product. Access is Utopia’s.
- Headroom memory, `headroom learn`, Headroom-as-knowledge.
- CCR markers, retrieve tools, durable originals, cross-request body caches (Release A).
- Pointing Utopia `chat_base_url` at Headroom.
- Plain hashes of tool arguments in the audit log.
- Claiming native shell/file/HTTP are audited.
- Publishing the Headroom container to the host.
