# graphgen-toolgrad

A bridge between two synthetic-data projects, built on Sri's integration
forks (branch `integrate/graphgen-toolgrad` on each):

- **[GraphGen](https://github.com/Srinivasoo7/GraphGen)** (fork of
  InternScience/GraphGen) — extracts a knowledge graph from your documents,
  then synthesizes SFT training data from it. The fork registers a new
  `trace_qa` operator: ToolGrad execution traces → tool-grounded QA pairs.
- **[ToolGrad](https://github.com/Srinivasoo7/toolgrad)** (fork of
  zhongyi-zhou/toolgrad, ACL'26 Findings) — answer-first tool-use dataset
  generation: build a *valid, executable* API chain first, then synthesize
  the user query and assistant response around it. The fork adds two public
  seams: an optional `kg_context` variable on `PREDICT_WORKFLOW` (threaded
  through `ToolGradState`), and a pluggable `sampler` on the MCP sampling
  path.

No monkeypatching: every integration point is a real, upstream-PR-able seam
in the forks.

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
    JSON/text), consumed via the fork's native `kg_context` seam.
  - Phase 2: `toolkg_builder.py` (ToolKG over the API catalog),
    `kg_sampler.py` (KG-neighborhood sampling), `toolkg_sampler.py`
    (builds the ToolKG from the fork's public `discover_mcp_tools` and
    returns a `sampler` callable for the fork's sampling seam).
  - Phase 3: `trace_to_qa.py` (trace → tool-grounded QA operator),
    `chain_verifier.py` (executable-chain verification). The operator is
    also registered natively in the GraphGen fork as
    `graphgen.operators.trace_qa.TraceQAService`.
  - Phase 4: `refinement_loop.py` (generate → verify → filter → refine,
    run ledger), `sft_mix.py` (SFT dataset assembly + dataset card),
    `eval_harness.py` (keyless dataset metrics).
  - `tests/` — keyless unit tests (`python3 bridge/tests/run_tests.py`),
    including `test_upstream_contract.py`, which pins the forks' seams
    and fails loudly if they ever move.

## The pipeline

```
GraphGen KG ──► kg_context ──┬──► PREDICT_WORKFLOW's {kg_context} (fork-native;
                             │         ToolGradState.kg_context)
API catalog ──► ToolKG ──────┴──► sampler(tools, num_apis) ──► get_mcp_apis
                                        │                        (fork seam)
ToolGrad loop (executes real chains) ───┘
        │
        ▼  ExecutionTracer / workflow samples
TraceToQAOperator / TraceQAService ──► tool-grounded QA pairs (ChatML)
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

# Python deps (pinned SHAs in pyproject.toml). This installs the bridge
# plus both integration forks from git — no separate upstream setup.
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the suite (no API keys, no Ray)
python3 bridge/tests/run_tests.py
```

Without the forks installed, the suite degrades gracefully: the runner
reports `SKIP` per module/test instead of crashing (bare stdlib: 1 passed,
15 skipped, 0 failed).

## Status

- Phase 0 (recon spike): done — `docs/phase0-recon.md`.
- Phase 1 (KG→ToolGrad bridge): done — `docs/phase1-bridge.md`.
- Phase 2 (ToolKG + guided sampling): done — `docs/phase2-toolkg.md`.
- Phase 3 (trace → tool-grounded QA): done — `docs/phase3-trace2qa.md`.
- Phase 4 (unified refinement/SFT/eval loop): done — `docs/phase4-unified-loop.md`.
- Ponytail revalidation: done — `docs/ponytail-review.md`.

Tests: **71/71 green**, keyless — `python3 bridge/tests/run_tests.py`.
Measured in the pinned environment (Python 3.12.3, `requirements.txt`
exact pins, ToolGrad fork @ `f4aaff10` installed editable); earlier "70/70"
and "74/74" claims predate the seam migration (the monkeypatch tests were
replaced by native seam/sampler tests).
The live end-to-end (real traces → real QA → SFT) needs `GOOGLE_API_KEY`;
the exact command sequence is in `docs/phase4-unified-loop.md`.

## Known limitations

- **The seams live on Sri's forks, not upstream.** `integrate/graphgen-toolgrad`
  on each fork is a small, upstream-PR-able diff (3 commits on toolgrad,
  1 on GraphGen), but until upstream merges equivalents, the bridge depends
  on the forks. `main` on both forks stays a clean upstream mirror.
  `bridge/tests/test_upstream_contract.py` pins the seam shapes and fails
  loudly if a fork rebase ever moves them.
- **GraphGen's Ray path is execution-unverified.** `TraceQAService` is a
  real registered operator (`graphgen.operators["trace_qa"]`, subclassing
  `BaseOperator`, standard `(results, meta_updates)` contract), and its
  `process()` was exercised directly — but `ray.init()` cannot complete in
  this sandbox, so the Ray pipeline path still needs a first real run.
- **Live LLM required for real data.** Everything here is validated
  keylessly on fixtures; generating actual KG-grounded tool-use data needs
  `GOOGLE_API_KEY` (or `OPENAI_API_KEY` / Vertex ADC) for ToolGrad's
  generation loop.

## Quick start (Phase 1 + 2, native seams)

```python
from toolgrad.modules import ToolGradState
from toolgrad.prebuilt import create_graph_on_mcp
from toolgrad.utils import mcp
from bridge import kg_context_exporter, toolkg_sampler

# 1. Load a GraphGen KG (networkx GraphML) and export domain context
graph = kg_context_exporter.load_from_working_dir("/path/to/kg")
kg_context = kg_context_exporter.render_kg_context(
    kg_context_exporter.export_kg_context(graph))

# 2. Build the ToolKG once, plug its sampler into the fork's seam
toolkg = toolkg_sampler.build_toolkg_for_mcp()
app = create_graph_on_mcp(
    sample_seed=123, num_apis=5, num_iterations=3,
    mcp_dict=mcp.get_default_mcp_dict(),
    api_sampler=toolkg_sampler.make_toolkg_sampler(toolkg),  # native seam
)

# 3. KG context travels in graph state — inverse_predictor grounds
#    queries/responses in the KG entities via the fork's {kg_context}
initial = ToolGradState(workflow_cur=None, api_proposals=None,
                        api_reports=None, api_selection=None, step=0,
                        sampled_apis=[], kg_context=kg_context)
final_state = app.invoke(initial)
```

Or, inside a GraphGen pipeline, use the registered operator directly:

```python
from graphgen.operators import operators
TraceQAService = operators["trace_qa"]
op = TraceQAService(working_dir="cache", kg_context=ctx_dict)
results, meta = op.process([{"sample": workflow_dict, "tracer": tracer_dict}])
```
