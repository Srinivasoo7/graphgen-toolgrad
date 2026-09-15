# graphgen-toolgrad

A bridge between two synthetic-data projects:

- **[GraphGen](https://github.com/InternScience/GraphGen)** — extracts a knowledge
  graph from your documents, then synthesizes SFT training data from it
  (atomic / multi-hop / multi-choice QA, fill-in-blank, CoT).
- **[ToolGrad](https://github.com/zhongyi-zhou/toolgrad)** (ACL'26 Findings) —
  answer-first tool-use dataset generation: build a *valid, executable* API
  chain first, then synthesize the user query and assistant response around it.
  Runs on ToolBench APIs or any MCP server.

## The problem this solves

There is no good way to generate training data for an agent that must **use
tools to answer questions grounded in a specific domain**.

- ToolGrad alone teaches generic function-calling: chains are randomly sampled
  from ToolBench, queries are plausible but detached from any real knowledge.
- GraphGen alone teaches answer-from-memory: QA is domain-grounded via the KG,
  but single-turn with no tool use.

Combined, the pipeline produces SFT examples where the query references **real
domain entities** (from the KG) and the answer can only be produced by
**executing a real, valid tool chain** — what a domain agent actually does.

## Layout

- `PLAN.md` — the phase-by-phase build plan (all four phases complete).
- `docs/` — per-phase recon notes and design docs.
- `bridge/` — adapter code:
  - Phase 1: `kg_context_exporter.py` (GraphGen KG → domain-context
    JSON/text), `prompts/predict_workflow_kg.py` (KG-grounded
    `PREDICT_WORKFLOW` template), `toolgrad_patch.py` (prompt patch).
  - Phase 2: `toolkg_builder.py` (ToolKG over the API catalog),
    `kg_sampler.py` (KG-neighborhood sampling), `toolkg_patch.py`
    (sampling patch).
  - Phase 3: `trace_to_qa.py` (trace → tool-grounded QA operator),
    `chain_verifier.py` (executable-chain verification).
  - Phase 4: `refinement_loop.py` (generate → verify → filter → refine,
    run ledger), `sft_mix.py` (SFT dataset assembly + dataset card),
    `eval_harness.py` (keyless dataset metrics).
  - `tests/` — keyless unit tests (`python3 bridge/tests/run_tests.py`).

## The pipeline

```
GraphGen KG ──► kg_context ──┬──► PREDICT_WORKFLOW (ToolGrad inverse predictor)
                             │         now grounds queries in real entities
API catalog ──► ToolKG ──────┴──► KG-neighborhood sampling (replaces random)
                                        │
ToolGrad loop (executes real chains) ───┘
        │
        ▼  ExecutionTracer / workflow samples
TraceToQAOperator ──► tool-grounded QA pairs (ChatML)
        │
chain_verifier + quality filters + textual-gradient refinement
        │
        ▼
sft_mix.jsonl (train/valid) + dataset card + eval metrics
```

## Status

- Phase 0 (recon spike): done — `docs/phase0-recon.md`.
- Phase 1 (KG→ToolGrad bridge): done — `docs/phase1-bridge.md`.
- Phase 2 (ToolKG + guided sampling): done — `docs/phase2-toolkg.md`.
- Phase 3 (trace → tool-grounded QA): done — `docs/phase3-trace2qa.md`.
- Phase 4 (unified refinement/SFT/eval loop): done — `docs/phase4-unified-loop.md`.

Tests: **70/70 green**, keyless — `python3 bridge/tests/run_tests.py`
(needs the ToolGrad venv + `networkx`; no API keys, no Ray).
The live end-to-end (real traces → real QA → SFT) needs `GOOGLE_API_KEY`;
the exact command sequence is in `docs/phase4-unified-loop.md`.

## Quick start (Phase 1)

```python
from bridge import kg_context_exporter, toolgrad_patch

# 1. Load a GraphGen KG (networkx backend GraphML) and export context
graph = kg_context_exporter.load_from_working_dir("/path/to/graphgen/working_dir")
ctx_text = kg_context_exporter.render_kg_context(
    kg_context_exporter.export_kg_context(graph))

# 2. Patch ToolGrad before building/invoking its graph
toolgrad_patch.apply_kg_patch(ctx_text)
# ... run the ToolGrad loop as usual; inverse_predictor now grounds
# queries/responses in the KG entities ...
toolgrad_patch.remove_kg_patch()
```
