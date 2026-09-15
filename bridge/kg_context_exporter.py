"""Export a GraphGen knowledge graph as compact domain context for ToolGrad.

Input is a :class:`networkx.Graph` — the same object held by GraphGen's
``networkx`` graph-storage backend. Two ways to obtain it without Ray:

1. From a live GraphGen run (networkx backend): pass
   ``storage.get_graph()`` directly (``GraphStorageActor`` proxy or the
   ``NetworkXStorage`` instance both expose it).
2. From a finished run: the networkx backend persists the KG as GraphML to
   ``<working_dir>/<namespace>.graphml`` (see ``NetworkXStorage.__post_init__``
   / ``index_done_callback``); load it with :func:`load_graphml` or
   :func:`load_from_working_dir`.

The exporter understands the node/edge attribute schema written by GraphGen's
``light_rag_kg_builder`` (``entity_name`` / ``entity_type`` / ``description``
on nodes; ``src_id`` / ``tgt_id`` / ``description`` on edges) but degrades
gracefully: any missing attribute falls back to the node id / a truncated
description, so hand-built graphs work too.

Output is a token-budgeted JSON document::

    {
      "entities": [{"name", "entity_type", "description", "degree"}],
      "triples":  [{"head", "relation", "tail", "description"}],
      "communities": [{"id", "size", "members", "summary"}],
      "meta": {...},
    }

plus :func:`render_kg_context`, which formats it as the compact text block
that is injected into ToolGrad's ``PREDICT_WORKFLOW`` prompt as
``{kg_context}``.

No Ray, no LLM, no API keys required.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import networkx as nx

# Attribute names written by GraphGen's light_rag KG builder
# (graphgen/models/kg_builder/light_rag_kg_builder.py). First hit wins.
_NODE_NAME_KEYS = ("entity_name", "name", "id")
_NODE_TYPE_KEYS = ("entity_type", "type")
_NODE_DESC_KEYS = ("description", "desc", "summary")
_EDGE_REL_KEYS = ("relation", "relationship", "edge_type", "relation_type", "type")
_EDGE_DESC_KEYS = ("description", "desc", "summary")

# Rough token estimate (char-based; ~4 chars/token for English prose).
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for budget enforcement."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _first_attr(attrs: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


@dataclass
class ExportConfig:
    """Knobs for the export. All caps are hard limits; ``max_tokens`` is a
    soft budget enforced after the caps by dropping lowest-priority items."""

    max_entities: int = 40
    max_triples: int = 60
    max_communities: int = 8
    max_tokens: int = 2000
    max_desc_chars: int = 300
    # Ranking: entities/triples are ranked by graph degree (most connected
    # first); ties broken by name for determinism.
    community_min_size: int = 2


def load_graphml(path: str) -> nx.Graph:
    """Load a GraphML file written by GraphGen's networkx storage backend."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GraphML file not found: {path}")
    return nx.read_graphml(path)


def load_from_working_dir(working_dir: str, namespace: str = "kg") -> nx.Graph:
    """Load ``<working_dir>/<namespace>.graphml``.

    ``namespace`` is the graph-storage namespace from the GraphGen run config
    (the file is ``f"{namespace}.graphml"`` in ``NetworkXStorage``).
    """
    return load_graphml(os.path.join(working_dir, f"{namespace}.graphml"))


def _ranked_entities(graph: nx.Graph, config: ExportConfig) -> list[dict]:
    degree = dict(graph.degree())
    # Deterministic: degree desc, then node id asc.
    ranked = sorted(graph.nodes(), key=lambda n: (-degree.get(n, 0), str(n)))
    entities = []
    for node in ranked[: config.max_entities]:
        attrs = dict(graph.nodes[node])
        entities.append(
            {
                "name": _first_attr(attrs, _NODE_NAME_KEYS, default=str(node)),
                "entity_type": _first_attr(attrs, _NODE_TYPE_KEYS, default="UNKNOWN"),
                "description": _truncate(
                    _first_attr(attrs, _NODE_DESC_KEYS), config.max_desc_chars
                ),
                "degree": int(degree.get(node, 0)),
            }
        )
    return entities


def _ranked_triples(graph: nx.Graph, config: ExportConfig) -> list[dict]:
    degree = dict(graph.degree())
    view = graph.to_undirected() if graph.is_directed() else graph
    edges = sorted(
        view.edges(data=True),
        key=lambda e: (-(degree.get(e[0], 0) + degree.get(e[1], 0)), str(e[0]), str(e[1])),
    )
    triples = []
    for u, v, attrs in edges[: config.max_triples]:
        attrs = dict(attrs)
        relation = _first_attr(attrs, _EDGE_REL_KEYS)
        description = _first_attr(attrs, _EDGE_DESC_KEYS)
        if not relation:
            # GraphGen's light_rag builder stores no explicit relation label;
            # the description carries the relationship.
            relation = _truncate(description, 80) or "related_to"
        triples.append(
            {
                "head": _first_attr(dict(view.nodes[u]), _NODE_NAME_KEYS, default=str(u)),
                "relation": relation,
                "tail": _first_attr(dict(view.nodes[v]), _NODE_NAME_KEYS, default=str(v)),
                "description": _truncate(description, config.max_desc_chars),
            }
        )
    return triples


def _communities(graph: nx.Graph, config: ExportConfig) -> list[dict]:
    view = graph.to_undirected() if graph.is_directed() else graph
    degree = dict(view.degree())
    components = [
        c for c in nx.connected_components(view) if len(c) >= config.community_min_size
    ]
    # Deterministic: size desc, then smallest member name asc.
    components.sort(key=lambda c: (-len(c), sorted(str(n) for n in c)[0]))
    out = []
    for i, comp in enumerate(components[: config.max_communities]):
        members = sorted(comp, key=lambda n: (-degree.get(n, 0), str(n)))
        member_names = [
            _first_attr(dict(view.nodes[n]), _NODE_NAME_KEYS, default=str(n))
            for n in members[:8]
        ]
        out.append(
            {
                "id": i,
                "size": len(comp),
                "members": member_names,
                "summary": (
                    f"Topic cluster of {len(comp)} entities, centered on "
                    + ", ".join(member_names[:5])
                    + (", …" if len(member_names) > 5 else "")
                    + "."
                ),
            }
        )
    return out


def _fit_budget(context: dict, config: ExportConfig) -> dict:
    """Enforce ``max_tokens`` by dropping lowest-priority items first.

    Priority (highest kept longest): entities > communities > triples.
    If a single remaining item still exceeds the budget, descriptions are
    truncated to fit.
    """
    context = {k: (list(v) if isinstance(v, list) else v) for k, v in context.items()}

    def over() -> bool:
        return estimate_tokens(render_kg_context(context)) > config.max_tokens

    while over() and context["triples"]:
        context["triples"].pop()
    while over() and context["communities"]:
        context["communities"].pop()
    while over() and len(context["entities"]) > 1:
        context["entities"].pop()
    if over():
        # Last resort: strip descriptions; names/relations alone are tiny.
        for entity in context["entities"]:
            entity["description"] = ""
        for triple in context["triples"]:
            triple["description"] = ""
        for community in context["communities"]:
            community["summary"] = (
                f"Topic cluster of {community['size']} entities."
            )
    return context


def export_kg_context(graph: nx.Graph, config: ExportConfig | None = None) -> dict:
    """Export ``graph`` to the budgeted domain-context document."""
    config = config or ExportConfig()
    context = {
        "entities": _ranked_entities(graph, config),
        "triples": _ranked_triples(graph, config),
        "communities": _communities(graph, config),
        "meta": {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
            "directed": bool(graph.is_directed()),
        },
    }
    context = _fit_budget(context, config)
    context["meta"].update(
        {
            "exported_entities": len(context["entities"]),
            "exported_triples": len(context["triples"]),
            "exported_communities": len(context["communities"]),
            "estimated_tokens": estimate_tokens(render_kg_context(context)),
            "max_tokens": config.max_tokens,
        }
    )
    return context


def render_kg_context(context: dict) -> str:
    """Render the exported context as compact text for the ``{kg_context}``
    prompt variable."""
    lines = ["Domain entities:"]
    for e in context.get("entities", []):
        desc = f": {e['description']}" if e.get("description") else ""
        lines.append(f"- {e['name']} ({e.get('entity_type', 'UNKNOWN')}){desc}")
    lines.append("")
    lines.append("Key relationships:")
    for t in context.get("triples", []):
        desc = f" — {t['description']}" if t.get("description") else ""
        lines.append(f"- {t['head']} --{t['relation']}--> {t['tail']}{desc}")
    lines.append("")
    lines.append("Topic clusters:")
    for c in context.get("communities", []):
        lines.append(
            f"- Cluster {c['id']} ({c['size']} entities): "
            f"{', '.join(c['members'])}. {c['summary']}"
        )
    return "\n".join(lines).strip()


def export_kg_context_json(graph: nx.Graph, config: ExportConfig | None = None) -> str:
    """JSON-serialized :func:`export_kg_context` (keys sorted for diffability)."""
    return json.dumps(export_kg_context(graph, config), indent=2, sort_keys=True,
                      ensure_ascii=False, default=str)
