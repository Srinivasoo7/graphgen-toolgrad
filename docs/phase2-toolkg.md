# Phase 2 — ToolKG: API knowledge graph + guided sampling

Date: 2026-09-15. All claims below are from code actually read and
commands actually run on this box (2 vCPUs, 7.7 GB RAM, no GPU).
`GOOGLE_API_KEY` / `OPENAI_API_KEY` are absent, so the live LLM loop is
**not** run here; the keyless build + tests are the deliverable.

## What was built

`bridge/` additions (no upstream code vendored; upstreams stay in
`~/workspace/vendor/` as reference only):

| File | Purpose |
|---|---|
| `bridge/toolkg_builder.py` | Builds a `networkx.DiGraph` over a tool catalog: nodes = tools (name, description, input/output properties), edges = composability (A's output plausibly feeds B's input), scored by a transparent name/type-overlap heuristic. Serializes to GraphML/JSON. |
| `bridge/kg_sampler.py` | `sample_api_neighborhood(toolkg, num_apis, ...)` — score-weighted random walk with restarts; replaces ToolGrad's uniform `random.sample`. Ships `uniform_sample` (baseline) and `neighborhood_density` (comparison metric). |
| `bridge/toolkg_patch.py` | `apply_toolkg_patch()` / `remove_toolkg_patch()`: runtime monkeypatch swapping `toolgrad.utils.mcp.get_mcp_apis` so `sample_apis_or_end` samples neighborhoods. No upstream edit. |
| `bridge/tests/test_toolkg_*.py` | 20 new unit tests; suite is now 35/35 green via `python3 bridge/tests/run_tests.py`. |

## The composability heuristic

For each ordered tool pair (A, B), A ≠ B:

1. **Outputs**: explicit `output_hints` per tool when the catalog owner
   knows what a tool returns; otherwise inferred from the tool description
   with a small keyword lexicon (`path`/`file`/`directory` → output `path`,
   `content`/`text` → output `content`, `listing` → `listing`, …).
2. **Inputs**: read from the tool's declared schema — handles langchain
   `StructuredTool` (`args_schema` as a plain JSON-schema dict *or* a
   pydantic model), plain `input_schema` dicts, and `args` dicts.
3. **Score**: best over all (output, input) property pairs of
   `Dice(token_set(out), token_set(in)) + 0.15` if the canonical types are
   compatible (string↔path, integer↔number, content↔string), capped at 1.0.
   Tokenization splits camelCase/snake_case and folds naive plurals
   (`paths`→`path`, `directories`→`directory`).
4. Edge A→B with `score`, `via="out->in"`, `relation="composable"` iff
   score ≥ threshold (default 0.34).

Worked example — the real MCP filesystem catalog (built live, keyless,
via npx; 5 tools, 8 edges, density 0.4, no isolates):

```
list_directory  --path->path (1.0)-->  read_text_file
directory_tree  --path->paths (1.0)--> read_multiple_files
read_text_file  --(no outgoing edges)   # its `content` output matches no input
```

The readers correctly have no outgoing edges (nothing consumes `content`
in this catalog), and the two listing tools feed all three readers —
exactly the composability structure a human would draw.

Node attributes follow GraphGen's networkx-backend conventions
(`entity_name`/`entity_type`/`description`), so the Phase-1
`kg_context_exporter` can render a ToolKG as domain context too.

## Limits of the heuristic (known, documented)

- **Description-inferred outputs are the weak link.** The lexicon guesses
  (`"listing"` → output `listing`) and will miss tools whose descriptions
  don't name their outputs. Prefer `output_hints` for real catalogs.
- **Name overlap ≠ semantic fit.** `path`→`paths` matches, but a tool
  returning `listing` (array of paths) won't match an input named `path`
  without hints — the plural fold only goes one level.
- **No type veto.** A perfect name match with incompatible types still
  creates an edge (just without the +0.15 bonus). The threshold is the
  only gate.
- **Static catalog.** The graph is built once per discovery; it doesn't
  learn from which chains actually execute. Execution feedback (Phase 4
  territory) could reweight edges.
- An LLM-based output-schema miner over API docs would be strictly
  better — but needs a key, so it's parked as a future enhancement.

## Where the sampler plugs into ToolGrad's loop

`toolgrad/modules/graph_lib.py::sample_apis_or_end` (line 44) calls
`mcp.get_mcp_apis(mcp_dict=..., num_apis=..., seed=...)` for the MCP path,
resolving `get_mcp_apis` on the `toolgrad.utils.mcp` module **at call
time**. The patch swaps that attribute with a wrapper that reuses
upstream's own discovery (`MultiServerMCPClient`, `ALLOWED_APIS` filter,
`_wrap_with_path_prefix`) and replaces only the final
`random.seed(seed); random.sample(tools, num_apis)` with
`sample_api_neighborhood`. The ToolBench path is untouched.

Determinism note: upstream reseeds the *global* RNG with `seed` on every
call, so every iteration samples the *same* bundle (likely an upstream
quirk). The wrapper uses a local `random.Random` seeded with a
deterministic fold of `(seed, call_count)` — same call sequence ⇒ same
samples, and iterations vary by default (`vary_per_call=False` restores
constant sampling). It also stops polluting global RNG state.

## What is tested (keyless, 35/35 green)

Run: `../vendor/.venv_toolgrad/bin/python bridge/tests/run_tests.py`
(from the repo root; needs the ToolGrad venv + `networkx`).

- **Builder** (10 tests): expected edges + scores + `via` labels on a
  6-tool hand-made catalog; no self-loops; directedness
  (`read_text_file → list_directory` correctly absent); threshold filters
  partial matches (0.817 edge drops at 0.9, exact matches survive);
  description-inference works with no hints; GraphGen node-attribute
  conventions; determinism; GraphML and JSON roundtrips.
- **Sampler** (6 tests): seeded determinism; seed-first/shape/uniqueness;
  score-weighting preference (200 trials: the 1.0 edge chosen >2× as often
  as the 0.35 edge); disconnected-component fallback fills the sample;
  **neighborhood vs uniform**: mean composability density 0.270 vs 0.110
  over 40 fixed-seed trials; error cases (empty graph, oversize request,
  unknown seed).
- **Patch** (4 tests): `graph_lib` resolves `mcp.get_mcp_api` via module
  attribute (the fact the patch relies on); apply → patched sampler
  returns real tool objects deterministically → remove restores the
  original; `num_apis` validation mirrors upstream's `ValueError`;
  double-apply idempotent; safe remove when unpatched.
- Phase 1's 15 tests still pass unchanged.

## What a live-LLM validation would measure

Once `GOOGLE_API_KEY` exists, run `examples/mcp_filesystem.py` twice —
once with `apply_toolkg_patch()` and once unpatched — and compare:

1. **Chain composability**: fraction of generated chains where ≥2
   consecutive tool calls correspond to a ToolKG edge (should rise
   sharply with the patch).
2. **Chain realism**: blind LLM-judge (or human) rating of whether each
   chain looks like a task a real user would pose, patched vs unpatched.
3. **Success rate**: fraction of chains that execute end-to-end without
   the selector/proposer bailing to the next iteration — composable
   bundles should fail less often.

The trace JSONs (`examples/outputs/trace/*.json`) already capture
everything needed; no new instrumentation required.

## Notes for Phase 3 (tool-grounded QA generator)

- `toolkg_patch._TOOLS_BY_NAME` maps sampled names back to live
  `StructuredTool` objects — a Phase 3 operator that needs the tools
  behind a chain can reuse this mapping (or rebuild it via
  `build_toolkg_for_mcp`, which caches).
- The `ExecutionTracer` JSONs contain the sampled API set per iteration
  (`sampled_apis`) plus the final chain; joining those names against the
  ToolKG gives Phase 3 a free "which composability edges were actually
  used" signal for ranking chains.
- `neighborhood_density` is a ready-made chain-quality feature for any
  Phase 3/4 reranking or filtering step.
