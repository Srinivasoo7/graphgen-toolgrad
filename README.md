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

- `PLAN.md` — the phase-by-phase build plan.
- `docs/` — per-phase recon notes and design docs.
- `third_party/` — pinned git submodules of the two upstream repos.
- `bridge/` — adapter code (lands starting Phase 1).

## Status

Phase 0 (recon spike) in progress — see `docs/phase0-recon.md` when it lands.
