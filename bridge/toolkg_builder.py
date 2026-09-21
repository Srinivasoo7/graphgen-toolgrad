"""Phase 2 — ToolKG: a knowledge graph over a tool/API catalog.

Nodes are tools (name, description, input/output properties). Edges are
*composability*: tool A's output plausibly feeds tool B's input, scored by a
transparent name/type-overlap heuristic (no LLM, no keys).

The produced :class:`networkx.DiGraph` uses the same node-attribute
conventions as GraphGen's networkx KG backend (``entity_name`` /
``entity_type`` / ``description``), so the Phase-1 ``kg_context_exporter``
can render a ToolKG as domain context too.

Why a heuristic instead of GraphGen's LLM extractor: MCP tools (and most
API catalogs) declare input schemas but *not* output schemas, so there is
nothing for an extractor to read off the schema. The builder therefore
accepts explicit ``output_hints`` per tool and otherwise falls back to
inferring likely outputs from the tool description with a small,
documented keyword lexicon. LLM-based output-schema mining from API docs
remains a possible (keyed) enhancement — see ``docs/phase2-toolkg.md``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

TOOLKG_VERSION = "0.1.0"
DEFAULT_THRESHOLD = 0.34
TYPE_BONUS = 0.15

# ---------------------------------------------------------------------------
# Tokenization / normalization helpers
# ---------------------------------------------------------------------------

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^a-zA-Z0-9]+")


def _singular(token: str) -> str:
    """Naive singularization: directories->directory, paths->path.

    Deliberately dumb and documented: irregular plurals and words like
    "metadata" pass through unchanged.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _tokens(name: Any) -> set:
    """Lowercased, singularized alphanumeric tokens of a property name."""
    parts = [p for p in _CAMEL_SPLIT.split(str(name)) if p]
    return {_singular(p.lower()) for p in parts}


_TYPE_ALIASES = {
    "str": "string", "string": "string", "text": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "double": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array",
    "dict": "object", "object": "object", "map": "object",
    "path": "path", "filepath": "path",
    "content": "content",
}


def _normalize_type(type_name: Any) -> str:
    """Map a JSON-schema-ish type name to a canonical type.

    Unknown / missing types default to ``"string"`` (the common case for
    tool parameters).
    """
    return _TYPE_ALIASES.get(str(type_name).strip().lower(), "string")


def _types_compatible(out_type: str, in_type: str) -> bool:
    if out_type == in_type:
        return True
    if {out_type, in_type} <= {"string", "path"}:
        return True
    if {out_type, in_type} <= {"integer", "number"}:
        return True
    if {out_type, in_type} == {"content", "string"}:
        return True
    return False


# ---------------------------------------------------------------------------
# Composability scoring
# ---------------------------------------------------------------------------


def name_similarity(a: Any, b: Any) -> float:
    """Sørensen–Dice coefficient over token sets: 2|∩| / (|a|+|b|)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


# Tokens so generic they carry almost no composability evidence on their own.
# A match counts as specific only when the shared tokens include something
# else: ticket_id->ticket_id shares {"ticket", "id"} (specific via "ticket"),
# while ticket_id->article_id shares only {"id"} (generic-only -> penalized).
_GENERIC_TOKENS = frozenset({
    "id", "ids", "title", "name", "names", "status", "state", "type", "kind",
    "description", "desc", "info", "data", "value", "result", "results",
    "item", "items", "list", "count", "total", "summary",
})


def composability_score(
    outputs: Sequence[Tuple[str, str]],
    inputs: Sequence[Tuple[str, str]],
) -> Tuple[float, Optional[Tuple[str, str]]]:
    """Best (output -> input) property match between two tools.

    Score = Dice name similarity + ``TYPE_BONUS`` when the types are
    compatible, capped at 1.0 — except when the shared tokens are all
    generic (``"id"``, ``"title"``, ``"status"``, ...): that is weak
    evidence of real composability, so the score is halved and the type
    bonus is skipped. Returns ``(score, (out_name, in_name))``; the pair
    is ``None`` when nothing matches.
    """
    best = 0.0
    best_pair: Optional[Tuple[str, str]] = None
    for out_prop in outputs:
        out_name, out_type = out_prop[0], out_prop[1]
        ta = _tokens(out_name)
        for in_prop in inputs:
            # inputs may be (name, type) or (name, type, description)
            in_name, in_type = in_prop[0], in_prop[1]
            tb = _tokens(in_name)
            shared = ta & tb
            if not shared:
                continue
            s = 2 * len(shared) / (len(ta) + len(tb))
            if shared <= _GENERIC_TOKENS:
                s *= 0.5
            elif _types_compatible(out_type, in_type):
                s = min(1.0, s + TYPE_BONUS)
            if s > best:
                best, best_pair = s, (str(out_name), str(in_name))
    return best, best_pair


# ---------------------------------------------------------------------------
# Tool normalization
# ---------------------------------------------------------------------------

# (description keywords -> inferred output property name, type).
# Keywords are singular: description tokens are singularized by _tokens().
_OUTPUT_LEXICON = [
    ({"path", "filepath", "file"}, "path", "path"),
    ({"directory", "folder"}, "path", "path"),
    ({"content", "text", "body"}, "content", "content"),
    ({"list", "listing"}, "listing", "array"),
    ({"tree"}, "tree", "object"),
    ({"metadata", "info", "information", "detail", "summary"}, "metadata", "object"),
]


def infer_outputs_from_description(description: str) -> List[Tuple[str, str]]:
    """Guess a tool's output properties from description keywords.

    This is the weakest link of the heuristic (documented in
    ``docs/phase2-toolkg.md``): prefer explicit ``output_hints`` whenever
    the catalog owner knows what a tool returns.
    """
    toks = _tokens(description or "")
    found: List[Tuple[str, str]] = []
    seen = set()
    for keywords, prop_name, prop_type in _OUTPUT_LEXICON:
        if toks & keywords and prop_name not in seen:
            seen.add(prop_name)
            found.append((prop_name, prop_type))
    return found


def input_properties(tool: Any) -> List[Tuple[str, str, str]]:
    """Extract ``(name, type, description)`` input properties from a tool.

    Handles langchain ``StructuredTool`` objects (``args_schema`` as a
    plain JSON-schema dict *or* a pydantic model), plain dicts with an
    ``input_schema`` key, and objects with an ``args`` dict.
    """
    schema: Optional[Mapping] = None
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, Mapping):
        schema = args_schema
    elif args_schema is not None and hasattr(args_schema, "model_json_schema"):
        try:
            schema = args_schema.model_json_schema()
        except Exception:  # noqa: BLE001 - be liberal in what we accept
            schema = None
    if schema is None:
        maybe = getattr(tool, "input_schema", None)
        if isinstance(maybe, Mapping):
            schema = maybe

    props: List[Tuple[str, str, str]] = []
    if isinstance(schema, Mapping):
        raw = schema.get("properties", {})
        if isinstance(raw, Mapping):
            for pname, pspec in raw.items():
                if isinstance(pspec, Mapping):
                    ptype = pspec.get("type", "string")
                    pdesc = pspec.get("description", "") or ""
                else:
                    ptype, pdesc = "string", ""
                props.append((str(pname), _normalize_type(ptype), str(pdesc)))
    elif isinstance(getattr(tool, "args", None), Mapping):
        for pname in tool.args:
            props.append((str(pname), "string", ""))
    return props


def normalize_tool(
    tool: Any,
    output_hints: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
) -> Dict[str, Any]:
    """Normalize one tool into a plain spec dict.

    ``output_hints`` maps tool name -> list of ``(prop_name, prop_type)``
    describing what the tool returns. When absent for a tool, outputs are
    inferred from the description lexicon.
    """
    name = getattr(tool, "name", None) or "unknown_tool"
    description = getattr(tool, "description", "") or ""
    inputs = input_properties(tool)
    if output_hints and name in output_hints:
        outputs = [(str(on), _normalize_type(ot)) for on, ot in output_hints[name]]
    else:
        outputs = infer_outputs_from_description(description)
    return {
        "name": str(name),
        "description": str(description),
        "inputs": inputs,    # [(name, type, description)]
        "outputs": outputs,  # [(name, type)]
    }


# ---------------------------------------------------------------------------
# Graph construction / persistence
# ---------------------------------------------------------------------------


def build_toolkg(
    tools: Sequence[Any],
    output_hints: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> nx.DiGraph:
    """Build the ToolKG over ``tools``.

    Edge A -> B (``relation="composable"``) exists when A's best output
    property matches one of B's input properties with
    ``composability_score >= threshold``. Edge attributes: ``score``,
    ``via`` (``"out_prop->in_prop"``).
    """
    specs = [normalize_tool(t, output_hints) for t in tools]
    g = nx.DiGraph()
    g.graph["toolkg_version"] = TOOLKG_VERSION
    g.graph["threshold"] = float(threshold)
    for spec in specs:
        g.add_node(
            spec["name"],
            entity_name=spec["name"],
            entity_type="tool",
            description=spec["description"][:2000],
            # JSON-encoded: GraphML attributes must be scalars.
            inputs_json=json.dumps(
                [{"name": n, "type": t, "description": d}
                 for n, t, d in spec["inputs"]]
            ),
            outputs_json=json.dumps(
                [{"name": n, "type": t} for n, t in spec["outputs"]]
            ),
            num_inputs=len(spec["inputs"]),
            num_outputs=len(spec["outputs"]),
        )
    for a in specs:
        for b in specs:
            if a["name"] == b["name"]:
                continue
            score, pair = composability_score(a["outputs"], b["inputs"])
            if score >= threshold and pair is not None:
                g.add_edge(
                    a["name"], b["name"],
                    relation="composable",
                    score=round(score, 3),
                    via=f"{pair[0]}->{pair[1]}",
                )
    return g


def save_toolkg(g: nx.DiGraph, path: str) -> str:
    """Persist a ToolKG to ``.graphml`` or ``.json`` (node-link format)."""
    path = str(path)
    if path.endswith(".graphml"):
        nx.write_graphml(g, path)
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(g), f, indent=2)
    else:
        raise ValueError(f"unsupported ToolKG suffix for {path!r} (use .graphml/.json)")
    return path


def load_toolkg(path: str) -> nx.DiGraph:
    """Load a ToolKG written by :func:`save_toolkg`."""
    path = str(path)
    if path.endswith(".graphml"):
        return nx.read_graphml(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return nx.node_link_graph(json.load(f))
    raise ValueError(f"unsupported ToolKG suffix for {path!r} (use .graphml/.json)")


def toolkg_stats(g: nx.DiGraph) -> Dict[str, Any]:
    """Small summary: size, density, mean out-degree, isolated tools."""
    n = g.number_of_nodes()
    out_deg = [d for _, d in g.out_degree()] if n else []
    return {
        "num_tools": n,
        "num_edges": g.number_of_edges(),
        "density": round(nx.density(g), 3),
        "avg_out_degree": round(sum(out_deg) / n, 2) if n else 0.0,
        "isolated_tools": sorted(nx.isolates(g.to_undirected())),
    }
