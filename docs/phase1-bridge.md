# Phase 1 — KG-grounded ToolGrad bridge

Date: 2026-09-14/15. All claims below are from code actually read and
commands actually run on this box (2 vCPUs, 7.7 GB RAM, no GPU).
`GOOGLE_API_KEY` / `OPENAI_API_KEY` are absent, so the live LLM loop is
**not** run here; the keyless build + tests are the deliverable.

## What was built

`bridge/` — the connection layer (no upstream code vendored; upstreams stay
in `~/workspace/vendor/` as reference only):

| File | Purpose |
|---|---|
| `bridge/kg_context_exporter.py` | Dumps a GraphGen KG (networkx object) to compact, token-budgeted domain-context JSON + prompt-ready text. No Ray, no LLM, no keys. |
| `bridge/prompts/predict_workflow_kg.py` | Version-controlled copy of ToolGrad's `PREDICT_WORKFLOW` template with a new `{kg_context}` variable + grounding instructions. The only diff vs upstream is one inserted block (marked `[KG-CONTEXT]`). |
| `bridge/toolgrad_patch.py` | `apply_kg_patch(ctx)` / `remove_kg_patch()`: runtime monkeypatch that swaps the template and rebinds the updater. No upstream edit required. |
| `bridge/tests/` | 15 unit tests, runnable keyless via `python3 bridge/tests/run_tests.py` (no pytest on this box). |

## Architecture of the connection

```
GraphGen (networkx backend)                ToolGrad (MCP path)
─────────────────────────────              ────────────────────
KG lives in NetworkXStorage                LangGraph loop:
  ├─ storage.get_graph()  ──┐                sample_apis → proposer
  └─ <wd>/<ns>.graphml    ──┤                → executor → selector
                            ▼                → ★ inverse_predictor ★
              bridge/kg_context_exporter
              {entities, triples,              │
               communities} → text             ▼
                            ──┤      PREDICT_WORKFLOW_KG
                                     {api_use_chains, kg_context}
                                            │
                                            ▼
                                   query/response grounded in
                                   real domain entities
```

The exporter reads the KG either live (`storage.get_graph()` from a
GraphGen run using the `networkx` backend) or from disk
(`load_from_working_dir(working_dir, namespace)` reads the
`<namespace>.graphml` that backend persists). It understands the
`light_rag_kg_builder` attribute schema (`entity_name`/`entity_type`/
`description` on nodes, `description` on edges — GraphGen's builder emits
no explicit relation label, so the exporter falls back to the edge
description) and degrades gracefully on hand-built graphs.

Budgeting: `ExportConfig(max_entities=40, max_triples=60,
max_communities=8, max_tokens=2000, max_desc_chars=300)`. Entities/triples
rank by graph degree (deterministic); communities are connected
components with auto summaries. `max_tokens` is enforced after the caps by
dropping triples → communities → entities (lowest priority first).

## Exact injection point

`toolgrad/modules/prompt_lib.py:153` — `PREDICT_WORKFLOW`, a
`ChatPromptTemplate` with a single variable `{api_use_chains}`, invoked by
`module_lib.create_workflow_updater()` (line 1162), which is called with no
args from `graph_lib.inverse_predictor` (line ~264) and then
`.invoke({"api_use_chains": chains_new})`.

The patch exploits two facts:
1. `create_workflow_updater` looks up `prompt_lib.PREDICT_WORKFLOW` via
   module attribute **at call time** → swapping the attribute swaps the
   template with no upstream edit.
2. `ChatPromptTemplate.partial(kg_context=...)` pre-binds the new variable,
   so the existing `invoke({"api_use_chains": ...})` call site in
   `graph_lib.py` needs no change.

Manual-edit alternative (if monkeypatching is ever undesirable): copy
`PREDICT_WORKFLOW_KG_TEXT` into `toolgrad/modules/prompt_lib.py`
(replacing `PREDICT_WORKFLOW`) and change the invoke call in
`graph_lib.py` to pass `kg_context`. The monkeypatch path is recommended
and is what the tests cover.

## What is tested (keyless, 15/15 green)

Run: `python3 bridge/tests/run_tests.py` (repo root; needs `networkx` and
the ToolGrad stack — the `~/workspace/vendor/.venv_toolgrad` venv, into
which `networkx` was pip-installed for this purpose).

- **Exporter** (`test_kg_context_exporter.py`, 8 tests): JSON shape,
  cap enforcement, token-budget enforcement, relation-label fallback,
  graceful handling of missing attributes and empty graphs.
- **Patch** (`test_toolgrad_patch.py`, 5 tests): template text contains
  exactly the two variables and no stray braces; renders with a fake
  context; monkeypatch apply/remove/idempotency against fake modules.
- **Trace shape** (`test_trace_shape.py`, 2 tests): a fabricated 2-step
  MCP-filesystem chain validates against the **real**
  `toolgrad.states.ApiUseWorkflow` pydantic model (keyless import) and
  round-trips through `model_dump()`/JSON — this pins the Phase 3 input
  contract. (Fixture note: the installed `langchain_classic`'s
  `ToolAgentAction` requires `message_log`/`tool_call_id`/`type` fields.)

Also verified live against the real ToolGrad modules (no LLM call):
`apply_kg_patch` swaps `prompt_lib.PREDICT_WORKFLOW` (vars become
`{api_use_chains, kg_context}`), the rendered prompt contains the KG
context, and `remove_kg_patch` restores the originals.

## What still needs a live LLM run

The full `app.invoke(...)` loop (`examples/mcp_filesystem.py` flow with the
patch applied): whether the grounded queries are actually better than the
unpatched baseline. That needs `GOOGLE_API_KEY` (or another provider key)
and a real KG (from a GraphGen run, or the exporter's hand-built path).

## Live end-to-end command (once `GOOGLE_API_KEY` is present)

Prerequisites: `~/workspace/vendor/.venv_toolgrad` (has `networkx`
installed 2026-09-15), a GraphGen KG as GraphML
(e.g. `<graphgen_working_dir>/kg.graphml` from a run with the `networkx`
backend — GraphGen itself needs an LLM key for extraction, or substitute
any GraphML), and `npx`/`node` for the MCP filesystem server.

```bash
cd ~/workspace/vendor/toolgrad
GOOGLE_API_KEY="<key>" ../.venv_toolgrad/bin/python - <<'EOF'
import sys, gin
sys.path.insert(0, "/home/hatch/workspace/graphgen-toolgrad")

# 1. KG -> domain context
from bridge import kg_context_exporter, toolgrad_patch
graph = kg_context_exporter.load_from_working_dir(
    "/path/to/graphgen/working_dir", namespace="kg")
context = kg_context_exporter.export_kg_context(graph)
print(kg_context_exporter.render_kg_context(context)[:500])

# 2. Patch ToolGrad BEFORE building the graph
toolgrad_patch.apply_kg_patch(kg_context_exporter.render_kg_context(context))

# 3. Gin LLM config + build graph (MCP path: no ToolBench key needed)
gin.parse_config_file("examples/configs/gemini-2.5-lite.gin")
import toolgrad as tog
app = tog.prebuilt.create_graph_on_mcp(
    sample_seed=123, num_apis=5, num_iterations=3,
    mcp_dict=tog.utils.mcp.get_default_mcp_dict())

# 4. Run (same as examples/mcp_filesystem.py)
tracer = tog.utils.trace_utils.ExecutionTracer(
    output_dir="examples/outputs/", seed=123)
initial = tog.modules.ToolGradState(
    workflow_cur=None, api_proposals=None, api_reports=None,
    api_selection=None, step=0, sampled_apis=[], tracer=tracer)
final_state = app.invoke(initial,
    config={"configurable": {"thread_id": 42}, "recursion_limit": 1000})
tracer.save()
sample = final_state["workflow_cur"]
print("QUERY:", sample.query)
print("RESPONSE:", sample.response[:300])
EOF
```

Compare `sample.query` against an unpatched run (skip step 2): the patched
query should name real entities from the KG instead of invented ones.
`examples/outputs/trace/00123.json` holds the execution trace (Phase 3
input); `seed=123__iter=3__num_apis=5.json` holds the sample.

## Notes for Phase 2 (ToolKG)

- The exporter's `_ranked_triples` already computes the primitive Phase 2
  needs: edges ranked by endpoint-degree sum. The ToolKG builder can reuse
  this module against an API-catalog graph (nodes = APIs, edges =
  composability).
- `toolgrad/modules/graph_lib.py::sample_apis_or_end` is the sampling
  function Phase 2 will replace with KG-neighborhood sampling; the patch
  pattern in `bridge/toolgrad_patch.py` shows how to swap it without
  editing the checkout.
- Watch: `ToolAgentAction` in the installed `langchain_classic` requires
  `message_log`/`tool_call_id` — any Phase 3 trace consumer constructing
  these models must include those fields (see the test fixture).
