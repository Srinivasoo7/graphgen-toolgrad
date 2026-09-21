# Enterprise ITSM live run — 2026-09-20

First end-to-end GraphGen–ToolGrad run against a realistic enterprise domain:
synthetic tool-use training data for a helpdesk copilot.

## Scenario

- MCP server: `itsm_mcp_server.py` — 7 tools: `search_kb`, `get_kb_article`,
  `create_ticket`, `get_ticket`, `update_ticket`, `list_open_tickets`,
  `lookup_asset`. Seeded with 12 KB articles (`corpus/`) and 5 assets.
- Config: `ggt-itsm.yaml` (6 chains requested, gemini-2.5-flash-lite via
  OpenRouter, `llm.max_tokens: 4096`).
- Full outputs: `runs/itsm/` (report, ledger, traces, SFT mix, eval).

## Measured result (`runs/itsm/run_report.json`)

- Status `ok`, 232.8s. 7 tools discovered. Corpus KG: 175 nodes / 154 edges.
- Generation: 5 workflows from 6 requested chains, 0 chain failures, 85 LLM
  calls. Refinement kept 1 of 5 samples. Dataset: 0 train / 1 validation.
- Kept pair (single step): Q "What is the status of the ticket INC-1042?" →
  A "The status of ticket INC-1042 is Open." (via `create_ticket`).
- Eval on the retained pair: chain-verified 1.0, ToolKG coverage 1.0,
  entity grounding 1.0. Secret scan: clean, zero findings.

## Honest limitations

- Yield is thin: the sole retained trajectory is a trivial single-step chain.
  Valuable multi-step paths (`search_kb → get_kb_article → create_ticket`)
  did not emerge.
- ToolKG inferred **0 composability edges** — this run supplied no
  `generation.output_hints`, so every tool stayed isolated.
- This is a plumbing proof in a plausible domain, not evidence of
  enterprise-scale readiness or downstream model improvement.

## Repro note

This run used a local `llm.max_tokens` fix (caps OpenRouter output so it no
longer pre-authorizes 65k tokens and 402s low-balance keys). Reproducing with
`ggt-itsm.yaml` needs that fix present in the bridge code.
