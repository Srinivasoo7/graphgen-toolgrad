# Token-savings measurement

This is a hypothesis, not a guarantee. Hold the corpus and questions constant.

Four cells:

| Cell | Knowledge | How it is recorded |
|---|---|---|
| A | Document-only (no graph assist) | `tokens_before` from the same authorized B ask |
| B | Document-only | On (`POST /v1/compress` after Utopia authorized the payload) |
| C | Graph-assisted (`entity_facts` via `POST /v1/ask`) | `tokens_before` from the same authorized D ask |
| D | Graph-assisted | On |

Cell B is `search_chunks`. Cell D is `find_entities` then `entity_facts` with that `entity_id`. If find_entities returns no id, C and D are skipped (`no_graph_coverage`) — do not treat a short error string as graph compression.

A and C are **not** a second HTTP call and **not** a caller bypass flag or header. They reuse `tokens_before` from the compress-on ask. Bypass remains trusted `CallKind` in `crux/compress.py` only.

Record `tokens_before`, `tokens_after`, and `tokens_saved` from the Crux ask response. Do not compare an empty knowledge base against a grown ontology and call that a compression result.

Graph-first only when coverage fits the question. Document fallback otherwise.
