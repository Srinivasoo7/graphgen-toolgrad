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

## Install (from zero)

```bash
git clone https://github.com/Srinivasoo7/graphgen-toolgrad
cd graphgen-toolgrad

# 1. Fetch the pinned upstreams (GraphGen @ 3a3eb097, ToolGrad @ c9544f84)
#    into ../vendor and pip-install them editable. Re-runnable.
bash scripts/setup_upstreams.sh

# 2. Python deps (pinned; measured green on Python 3.12.3)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # networkx + langchain-core + langchain-mcp-adapters
pip install -e .                  # the bridge package itself

# 3. Run the suite (no API keys, no Ray)
python3 bridge/tests/run_tests.py
```

Without the optional deps the suite degrades gracefully: the runner reports
`SKIP` per module/test instead of crashing (bare stdlib: 3 passed, 17
skipped, 0 failed). Without ToolGrad installed, the patch modules and the
upstream contract test skip — everything else still runs.

## Status

- Phase 0 (recon spike): done — `docs/phase0-recon.md`.
- Phase 1 (KG→ToolGrad bridge): done — `docs/phase1-bridge.md`.
- Phase 2 (ToolKG + guided sampling): done — `docs/phase2-toolkg.md`.
- Phase 3 (trace → tool-grounded QA): done — `docs/phase3-trace2qa.md`.
- Phase 4 (unified refinement/SFT/eval loop): done — `docs/phase4-unified-loop.md`.
- Ponytail revalidation: done — `docs/ponytail-review.md`.

Tests: **74/74 green**, keyless — `python3 bridge/tests/run_tests.py`.
Measured in the pinned environment (Python 3.12.3, `requirements.txt`
exact pins, ToolGrad @ `c9544f84` installed editable); earlier docs saying
"70/70" predate the 4 upstream-contract tests added in this pass.
The live end-to-end (real traces → real QA → SFT) needs `GOOGLE_API_KEY`;
the exact command sequence is in `docs/phase4-unified-loop.md`.

## Known limitations

- **Runtime monkeypatching is the integration mechanism** — deliberate and
  documented, but fragile: `toolgrad_patch` swaps `PREDICT_WORKFLOW` /
  `create_workflow_updater`, and `toolkg_patch` swaps `get_mcp_apis` and
  calls the *private* `toolgrad.utils.mcp._wrap_with_path_prefix`. A ToolGrad
  refactor of those call sites will break the bridge.
  `bridge/tests/test_upstream_contract.py` is the tripwire: re-run the suite
  after any upstream SHA bump and it fails loudly on exactly what moved.
- **GraphGen is format-compatible, not plugged in.** The bridge reads
  GraphGen's networkx GraphML output and emits ChatML/ShareGPT-shaped rows,
  but `TraceToQAOperator` is not registered in GraphGen's Ray engine. The
  registration path (needs a Ray-capable environment — `ray.init()` cannot
  complete in this sandbox): 1) subclass `graphgen.bases.BaseOperator`
  (the operator already mirrors its `process(batch) -> (results, stats)`
  shape); 2) place it under `graphgen/operators/generate/`; 3) wire it into
  a Ray pipeline script.
- **Live LLM required for real data.** Everything here is validated
  keylessly on fixtures; generating actual KG-grounded tool-use data needs
  `GOOGLE_API_KEY` (or `OPENAI_API_KEY` / Vertex ADC) for ToolGrad's
  generation loop.

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
