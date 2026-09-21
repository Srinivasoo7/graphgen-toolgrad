# graphgen-toolgrad factory — product guide

A factory that turns **your tool surface** (any MCP servers) into **verified,
knowledge-grounded tool-use QA data** for SFT — with chain execution
verification, a ToolKG composability graph, refinement, PII redaction, and a
secret sweep on every run.

```
MCP servers → discovery → domain KG → ToolKG → ToolGrad generation
     → refinement (quality gates + optional Jev semantic filter)
     → SFT mix + dataset card → eval harness → run report
```

No upstream code is vendored. The factory drives Sri's forks
(`Srinivasoo7/toolgrad`, `Srinivasoo7/GraphGen`) through their native seams.

## Quickstart

```bash
python -m bridge.cli init --name acme-crm --output ggt.yaml
# edit ggt.yaml: point mcp_servers at your tools, pick domain.kg_source
python -m bridge.cli validate-config ggt.yaml
python -m bridge.cli run --config ggt.yaml
python -m bridge.cli report ./runs/acme-crm
```

A run directory contains: `checkpoints/`, `events.jsonl`, `run.log`,
`ledger.json`, `kept.json`, `sft_mix/`, `DATASET_CARD.md`, `run_report.json`.
Resume an interrupted run with `run --config ggt.yaml --resume`.

Gate semantics worth knowing:

- **ToolKG coverage**: fraction of consecutive tool pairs that are ToolKG
  edges. A single-step chain scores 1.0 (vacuously composable) — simple
  lookups aren't penalized for having no pairs to judge.
- **Entity grounding**: deterministic, no LLM — a KG entity counts as
  referenced when its exact name appears in a step's tool name, inputs, or
  result preview, and the question names it. When a rewrite drops grounding,
  the critique names the exact candidate entities so the LLM can comply.
- **Stale checkpoints**: every checkpoint is stamped with a content hash of
  the bridge code. On resume, a checkpoint written by different code is
  re-run, not silently reused (logged as `stale_checkpoint` in
  `events.jsonl`). The refine summary reports the gate breakdown
  (`num_chain_valid`, `num_entity_grounded`, `mean_coverage`) so an empty
  result is diagnosable from `run_report.json`.

## Four enterprise patterns

### 1. Filesystem / ops tools (zero-LLM grounding)

Tool schemas alone ground the KG — no docs, no LLM needed.

```yaml
mcp_servers:
  - name: ops
    transport: stdio
    command: /opt/acme/bin/mcp-ops-server
    timeout_s: 120.0
domain:
  name: ops
  kg_source: tools
```

### 2. CRM / API surface (document-grounded)

Drop API docs, runbooks, or ticket exports (markdown/text) in a directory;
the factory extracts a KG from them with the configured LLM.

```yaml
domain:
  name: crm
  kg_source: corpus
  corpus_dir: ./docs/crm
  max_entities: 500
```

### 3. Regulated domain (explicit spec)

Compliance-heavy domains can hand-author the KG — no LLM extraction,
fully auditable.

```yaml
domain:
  name: claims
  kg_source: spec
  spec_path: ./spec/claims_kg.yaml
```

`claims_kg.yaml` format:

```yaml
entities:
  - {name: Claim, type: concept}
  - {name: Policy, type: concept}
relations:
  - {head: Claim, relation: filed_under, tail: Policy}
```

### 4. Semantic quality gate (Jev second-stage filter)

Heuristic gates (chain-valid, entity-grounded, ToolKG coverage) run first;
Jev judges semantic answer quality on the survivors.

```yaml
quality:
  semantic_filter: jev
  jev_cli: /opt/jev/bin/jev   # required when semantic_filter: jev
```

Measured on a 12-pair A/B (jev-1.13.0): 9/12 agreement with heuristics; Jev
rejected empty, contradictory, and fluff answers the heuristics accepted.
No verdict flips across reruns. Use as a second stage, not the sole gate.
See `docs/jev-filter-spike.md`.

## Credentials

The factory never takes a plaintext key. Two supported paths:

- **Enterprise deploy:** set the env var named by `llm.api_key_env`
  (default `OPENROUTER_API_KEY`).
- **This platform:** use the connected `custom.openrouter` credential —
  requests carry an authd surrogate exchanged at egress; the raw key never
  enters the process, logs, or disk.

Generation (`toolgrad_gen`) and refinement (`runner`) resolve credentials the
same way. Cost control is split honestly by path:

- `generation.max_llm_calls` (0 = uncapped) soft-caps LLM calls inside ToolGrad
  generation: checked after every chain outcome, remaining chains are
  skipped once exceeded, but the in-flight chain always finishes so the
  count can overshoot by up to one chain's calls.
- `llm.budget_usd` (0 = uncapped) aborts the LLMClient path — KG extraction
  from a corpus and refinement critiques — before the next call.
- `generation.min_samples` fails the run when generation yields too little
  (default 1): a run that produces zero usable chains is never reported
  "ok". Per-chain failures ship in `run_report.json`'s `generate` summary.

## Safety rails (every run)

- **PII redaction** on rendered contexts (emails, phones, SSNs, keys/tokens).
- **Secret scan** over final artifacts — the run fails if a private key,
  cloud credential, or bearer token reaches the outputs.
- **Chain verification** re-executes every tool chain (async-aware) before a
  pair can be kept. Discovery and every later tool call share one event
  loop for the whole run, and stdio MCP servers inherit proxy/CA env vars
  (without them `npx`-spawned servers retry their registry fetch until the
  timeout fires — which looks exactly like a hung session). Every
  verification call has a 60s timeout, so a stuck tool becomes a failed
  step, never a dead run.
- **Failure reports**: a crashed run still writes `run_report.json` with
  `failed_stage`, so CI can route on it.

## Config reference

See the scaffolded `ggt.yaml` comments for every field. The interesting
knobs:

| section | key | effect |
|---|---|---|
| `generation` | `num_chains`, `apis_per_workflow`, `num_iterations`, `seed` | dataset size/shape |
| `generation` | `min_samples`, `max_llm_calls` | yield floor (fail below) and LLM-call budget for generation |
| `generation` | `output_hints` | tool outputs the schema doesn't declare (feeds ToolKG edges) |
| `refinement`/`quality` | `min_coverage`, `require_chain_valid`, `require_entities` | structural gates |
| `quality` | `max_iterations` | textual-gradient refinement rounds (LLM cost) |
| `llm` | `model`, `max_rpm`, `budget_usd` | cost control |
| `sft` | `valid_ratio`, `max_pairs` | train/valid split |
