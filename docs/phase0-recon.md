# Phase 0 recon — GraphGen + ToolGrad bridge

Date: 2026-09-14/15. Box: 2 vCPUs, 7.7 GB RAM, no GPU, Ubuntu 24.04, Python 3.12.
All claims below are from code actually read and commands actually run on this
box. Nothing is asserted from memory or docs alone.

## Pinned upstream SHAs

| Repo | SHA (main) | Date |
|---|---|---|
| InternScience/GraphGen | `3a3eb097318d34b07f6f31cb60de5b20712d65ce` | 2026-08-17 |
| zhongyi-zhou/toolgrad | `c9544f842aacc13b5e44d5833fcc5dc4b86c5161` | 2026-06-11 |

Cloned to `~/workspace/vendor/GraphGen` and `~/workspace/vendor/toolgrad`
(shallow, 50 commits). Each has its own venv:
`~/workspace/vendor/.venv_graphgen`, `~/workspace/vendor/.venv_toolgrad`.

Note: PLAN.md suggested git submodules under `third_party/`; per the Phase 0
task instructions the bridge repo keeps upstream code out, so pins are
recorded here instead. `third_party/` stays empty for now.

---

## ToolGrad recon

### Architecture (read, not run, except where noted)

- Entry: `examples/mcp_filesystem.py` (MCP path) or `examples/toolbench.py`
  (ToolBench path).
- `toolgrad/prebuilt/toolgrad_on_mcp.py::create_graph_on_mcp` builds a
  LangGraph `StateGraph` with nodes:
  `sample_apis → api_proposer → api_executor → api_selector →
  inverse_predictor`. Verified: graph **builds** fine on this box
  (`CompiledStateGraph`, all 6 nodes present) with zero LLM calls.
- Core loop lives in `toolgrad/modules/graph_lib.py`; prompts in
  `toolgrad/modules/prompt_lib.py`; LLM backend in
  `toolgrad/utils/langchain.py::create_llm` (gin-configurable).
- **Phase 1 injection point (confirmed):** `PREDICT_WORKFLOW` in
  `toolgrad/modules/prompt_lib.py:153` is a `ChatPromptTemplate` taking
  `{api_use_chains}` and producing `{query, response}` via
  `module_lib.create_workflow_updater()` (structured output as
  `states.WorkflowQueryResponse`). Adding a `{kg_context}` variable here is
  the smallest real connection.
- MCP tool source: `toolgrad/utils/mcp.py::get_default_mcp_dict` spawns
  `npx -y @modelcontextprotocol/server-filesystem <dir>` over stdio, then
  filters to `ALLOWED_APIS = [read_file, list_directory, read_text_file,
  directory_tree, read_multiple_files]`.

### Dependencies

From `pyproject.toml`: `langgraph`, `langchain`, `langchain-core`,
`langchain-openai`, `langchain-community`, `langchain-google-genai`,
`langchain-google-vertexai`, `langchain-mcp-adapters`, `gin-config`,
`pydantic`, `anthropic`, `google-auth`, `huggingface-hub`, `numpy`, `pytz`.
`pip install -e .` succeeded (231 packages, exit 0).

### Required env vars / keys

LLM backend is chosen by the gin config (`examples/configs/*.gin`):

| gin `provider` | model example | required env |
|---|---|---|
| `google` (default in `gemini-2.5-lite.gin`) | `gemini-2.5-flash-lite` | `GOOGLE_API_KEY` (Google AI Studio; free tier exists) |
| `openai` | `gpt-*` | `OPENAI_API_KEY` |
| `azure` | `gpt-*` | `AZURE_O3_MINI_ENDPOINT`, `AZURE_O3_MINI_API_KEY` |
| `vertex` | `gemini-2.5-flash*` | `GOOGLE_CLOUD_PROJECT` + Application Default Credentials |

ToolBench path additionally needs `TOOLBENCH_KEY` (takes time to be issued)
and `TOOLBENCH_LIBRARY_ROOT`. The MCP path needs **no** ToolBench key —
only `npx`/`node` (present: node v24.20.0).

### What actually ran on this box

1. `pip install -e .` → OK.
2. `import toolgrad` → OK.
3. `create_graph_on_mcp(...)` → OK (graph builds, no LLM call).
4. `mcp.get_mcp_apis(...)` → **OK: 5 tools discovered**
   (`read_file`, `directory_tree`, `read_multiple_files`, `read_text_file`,
   `list_directory`). First attempt failed with `McpError: Connection closed`
   because `npx` was still downloading the server package; retry after the
   package cached succeeded. (Standalone `npx -y
   @modelcontextprotocol/server-filesystem` also starts cleanly.)
5. **Full `app.invoke(...)` data-generation loop → BLOCKED.** Every LLM call
   in the loop goes through `create_llm()`, which needs one of the keys
   above. `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `TOOLBENCH_KEY` are all absent
   from this environment. Per ground rules, no key was hunted for.

### Output schemas (from code)

- Per-run sample: `examples/outputs/seed={seed}__iter={iter}__num_apis={n}.json`
  = `ApiUseWorkflow.model_dump()`:
  ```json
  {
    "query": "user query synthesized from the chain",
    "api_use_chains": {
      "<chain_id>": {
        "chain_id": "...",
        "intermediate_steps": [[[tool, tool_input], result], ...],
        "description": "..."
      }
    },
    "response": "assistant response synthesizing chain results"
  }
  ```
- Execution trace: `examples/outputs/trace/{seed:05d}.json` via
  `ExecutionTracer`: `{seed, total_iterations, iterations:
  [{iteration, proposals, executions, selection, workflow_update}]}` —
  this is the Phase 3 input (chain traces → GraphGen generator).
- SFT rows (ToolGrad-500 style, `src/data/data_format_lib.py`):
  `{"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}`.

---

## GraphGen recon

### Architecture (read, not run end-to-end — see blockers)

- Entry: `python -m graphgen.run --config_file <yaml>`.
- `graphgen/engine.py::Engine` runs a declarative Ray DAG of `Node`s
  (`read → chunk → build_kg → partition → generate → …`), with Ray actors
  for the synthesizer/trainee LLMs and for KV/graph storage.
- Operator registry: `graphgen/operators/__init__.py`. Generators:
  `atomic`, `multi_hop`, `multi_choice`, `multi_answer`, `fill_in_blank`,
  `masked_fill_in_blank`, `cot`, `aggregated` (+ `vqa`, `quiz`).
- Output formats (`bases/base_generator.py::format_generation_results`):
  `Alpaca` → `{"instruction","input":"","output"}`;
  `Sharegpt` → `{"conversations":[{"from":"human"|"gpt","value"}]}`;
  `ChatML` → `{"messages":[{"role","content"}]}`;
  `QA_pairs` → `{"question","answer"}`.
  With `save_output: true`, JSONL lands under
  `<working_dir>/output/<unique_id>/`.
- Storage backends: graph `kuzu`/`networkx`; KV `rocksdb`/`json_kv`.
- **Phase 1 read point (confirmed):** KG lives behind
  `GraphStorageActor` / `init_storage(backend, working_dir, namespace)`;
  `networkx` backend keeps the graph in-process and is trivially dumpable
  to `{entities, relations, communities}` JSON for the ToolGrad prompt.

### Dependencies

`requirements.txt`: `ray[default]==2.53.0`, `pyarrow`, `kuzu`, `rocksdict`,
`networkx`, `leidenalg`, `igraph`, `tiktoken`, `openai`, `python-dotenv`,
`pyyaml`, `pandas`, `gr
...[truncated 5827 chars]