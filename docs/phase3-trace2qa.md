# Phase 3 — Tool-grounded QA generator

Date: 2026-09-15. All claims below are from code actually read and
commands actually run on this box (2 vCPUs, 7.7 GB RAM, no GPU).
`GOOGLE_API_KEY` / `OPENAI_API_KEY` are absent, so the live LLM loop is
**not** run here; the keyless build + tests are the deliverable.

## The idea

Phases 1–2 push domain knowledge *into* ToolGrad's generation loop.
Phase 3 pulls *verified tool-use data out*: ToolGrad's executed chains
(real tool calls with real results) are the one thing GraphGen cannot
produce on its own — GraphGen synthesizes QA from KG text but never
executes anything. The Phase 3 operator converts each executed chain into
an agentic QA pair whose question names real KG entities and is answerable
**only** by running the recorded tool chain (not from parametric
knowledge), in GraphGen's ChatML SFT format — the training data neither
project makes alone.

## What was built

| File | Purpose |
|---|---|
| `bridge/trace_to_qa.py` | `TraceToQAOperator`: extracts executed chains from ToolGrad workflow samples, anchors them to KG entities (deterministic), and emits `{question, answer_draft, required_tools, chain, entity_refs, provenance}` records. `llm_fn(prompt) -> str` seam; `to_chatml` for the SFT format. |
| `bridge/chain_verifier.py` | `verify_qa(qa, executor)`: checks every chain step — tool exists, input matches declared schema, execution doesn't raise. `MockToolExecutor` (keyless) + `ToolGradExecutor` (live, wraps real langchain tools). `chain_toolkg_coverage` scores chain↔ToolKG agreement. |
| `bridge/tests/test_trace_to_qa.py` | 8 tests |
| `bridge/tests/test_chain_verifier.py` | 8 tests |
| Suite | **51/51 green** via `../vendor/.venv_toolgrad/bin/python bridge/tests/run_tests.py` (repo root) |

## How it mirrors GraphGen's operator conventions

Studied `graphgen/operators/quiz/quiz_service.py` and
`graphgen/operators/generate/generate_service.py`:

1. **`process(batch) -> (results, stats)`** — both GraphGen operators take
   a batch of records and return results plus a stats dict. `TraceToQAOperator.process`
   does the same: `(samples, tracers) -> (qa_pairs, {"num_samples", "num_chains", "num_qa_pairs", ...})`.
2. **LLM behind an injected client** — GraphGen resolves `init_llm("synthesizer")`
   in `__init__`; here the constructor takes `llm_fn(prompt) -> str`, so
   tests inject a stub and the live run passes a real backend. When
   `llm_fn` is `None`, a deterministic template generator produces
   mechanical-but-honest drafts (grounded in matched entity names and tool
   names), keeping the whole pipeline keyless.
3. **ChatML SFT output** — GraphGen's generators emit ChatML
   (`data_format="ChatML"`). `to_chatml(qa)` emits
   system/user/assistant turns: the user turn carries the question +
   available tools, the assistant turn replays the executed chain and
   gives the drafted answer. Phase 4 consumes this directly.
4. **Prompt/parse separation** — `build_prompt` / `parse_llm_output` are
   separate methods (like GraphGen's `QuizGenerator.build_prompt_for_description`
   + `parse_rephrased_text`), so the prompt format is unit-testable without
   an LLM.

One deliberate deviation: GraphGen operators pull from storage backends
(`init_storage`); this operator takes plain dicts / JSON paths instead, so
it needs neither Ray nor rocksdb — the trace files on disk are the
interface.

## Input contracts (verified against real upstream code)

- **Primary**: `ApiUseWorkflow.model_dump()` JSON — the per-run sample
  file (`seed={s}__iter={n}__num_apis={k}.json`). Schema pinned by the
  Phase 1 test `test_trace_shape` against the real `toolgrad.states`
  pydantic models. `extract_chains` normalizes each chain's
  `intermediate_steps` to `[{tool, tool_input, result_preview}]` and
  skips chains with no executed steps.
- **Provenance (optional)**: `ExecutionTracer.save()` JSON
  (`{"seed", "total_iterations", "iterations"}`) — supplies `trace_seed`.
- **KG context**: Phase 1's `export_kg_context` dict. Entity anchoring is
  deterministic: an entity counts as referenced iff its exact name
  (case-insensitive) appears in a tool input value or result preview.
  No LLM involved in grounding.

## The verifier

`verify_qa` runs each chain step through three gates:
1. **tool exists** in the executor → else `missing_tools`,
2. **input schema check** (required properties + coarse JSON types) →
   else `schema_mismatches` with per-step problems,
3. **execute it** → exceptions are recorded per step (`RuntimeError: ...`),
   never raised.

`MockToolExecutor` registers `{name, input_schema, handler}` triples —
the whole suite runs keyless. `ToolGradExecutor` wraps real
`StructuredTool`s (`tool.invoke(input)`, same call the executor agent
makes); it reads schemas from pydantic `args_schema` or plain dicts, and
can be fed `toolkg_patch._TOOLS_BY_NAME`. `chain_toolkg_coverage` gives
the fraction of consecutive chain pairs that are ToolKG edges (1.0 on the
worked example) — a ready-made chain-quality feature for Phase 4
reranking.

## Worked example (actually run)

Fabricated-but-schema-faithful Tesla chain (2 MCP filesystem steps) +
2-entity KG + 6-tool ToolKG:

```
Q: Using the list_directory, read_text_file tools in order, what do the
   local records say about Tesla, Gigafactory Texas?
refs: ['Tesla', 'Gigafactory Texas']   tools: ['list_directory', 'read_text_file']
provenance: chain_0, seed 123, toolkg_edges_used=1, kg_entity_count=2
verify: True | toolkg coverage: 1.0 | chatml: [system, user, assistant]
```

## Limits (known, documented)

- **Mechanical drafts are shallow.** With `llm_fn=None` the question is a
  template ("what do the local records say about X?") — fine for plumbing,
  not for training data. Real quality needs the LLM pass.
- **Entity matching is substring-based.** "Texas" would also match
  "Gigafactory Texas"; overlapping names need longest-match ranking in
  Phase 4.
- **Answer drafts don't cite specific results.** The draft points at the
  chain; a refinement pass should quote the exact result preview.
- **One pair per chain by default.** `max_pairs_per_chain` exists for
  diversity, but true diversity (different questions per chain) needs the
  LLM with temperature.

## What the live run needs

`GOOGLE_API_KEY` (or another provider key), then:

1. Generate real traces: run the Phase 1/2 live commands (patched
   `PREDICT_WORKFLOW` + ToolKG sampling) to produce real
   `seed=*.json` sample files and `trace/*.json` tracer files.
2. Build a real KG with GraphGen (needs its own LLM key) and export via
   `kg_context_exporter`.
3. Run the operator with a real backend:

```python
from bridge import trace_to_qa
op = trace_to_qa.TraceToQAOperator(
    llm_fn=lambda p: llm.generate(p),   # your backend here
    kg_context=ctx, toolkg=toolkg)
pairs, stats = op.process([trace_to_qa.load_workflow_sample("seed=00123__iter=3__num_apis=5.json")],
                          [trace_to_qa.load_tracer("trace/00123.json")])
```

4. Verify every pair with `chain_verifier.verify_qa` + `ToolGradExecutor`
   before it enters the SFT mix — discard `chain_valid == False`.

## Notes for Phase 4 (unified loop)

- `chain_toolkg_coverage` + `neighborhood_density` (Phase 2) are two
  complementary chain-quality features: one checks the *sampled* chain
  against the ToolKG, the other the *catalog neighborhood* it came from.
  Use both for filtering.
- The verifier's per-step schema checks catch exactly the class of
  "hallucinated argument" errors that Phase 4's textual-gradient critic
  would otherwise have to rediscover.
- The ChatML output of `to_chatml` is the merge point with GraphGen's
  existing SFT pairs (same format, same fine-tuning recipe).
