# Build plan — all four phases complete (2026-09-15)

Keyless builds + tests are done and pushed; the live LLM end-to-end
(real traces → real QA → SFT) is documented in
`docs/phase4-unified-loop.md` and awaits `GOOGLE_API_KEY`.

## Phase 0 — Recon spike
- Pin both upstreams by recording their commit SHAs in `docs/phase0-recon.md`
  (upstreams live as local checkouts only — no submodules, no vendored code
  in this repo).
- Run ToolGrad's `examples/mcp_filesystem.py` end-to-end (MCP path needs no
  ToolBench API key; works CPU-only).
- Run GraphGen's quickstart on a tiny corpus.
- Document: exact dependency sets, required env vars / API keys, output
  schemas (ToolGrad chain JSON, GraphGen SFT JSONL), blockers.
- Deliverable: `docs/phase0-recon.md`.

## Phase 1 — KG-grounded ToolGrad (smallest real connection)
- `bridge/kg_context_exporter.py`: dump GraphGen's KG (entities, relations,
  communities) into compact domain-context JSON.
- Patch ToolGrad's `inverse_predictor` prompt template to consume it, so
  synthesized user queries reference real domain entities.
- Deliverable: N sample chains whose queries are KG-grounded; before/after
  comparison doc.

## Phase 2 — ToolKG (API knowledge graph)
- Build a KG over the API catalog itself: nodes = APIs, edges = composability
  (A's output schema fits B's input), using GraphGen's extractor + storage.
- Replace ToolGrad's random `sample_apis` with KG-neighborhood sampling.
- Deliverable: more realistic multi-hop chains; chain-realism eval notes.

## Phase 3 — Tool-grounded QA generator (GraphGen operator)
- New GraphGen generator that consumes ToolGrad execution traces
  (`ExecutionTracer` JSON) and emits (question, tool chain, answer) triples in
  GraphGen's SFT format.
- Deliverable: agentic-QA SFT data neither project makes alone.

## Phase 4 — Unified loop
- Port ToolGrad's textual-gradient refinement loop as a critic around
  GraphGen's generators.
- Merge both outputs into one SFT mix; fine-tune a small model with
  ToolGrad's SFT recipe; eval on BFCL + a knowledge-QA probe.
- Deliverable: trained-model eval report.

## Ground rules
- Never commit secrets (API keys, tokens). Env vars documented, values stay
  local.
- Upstream pins are recorded SHAs in `docs/phase0-recon.md`; upgrades are
  explicit doc updates. Upstream code is never vendored into this repo.
- Both upstreams are Apache 2.0; this repo is Apache 2.0.
