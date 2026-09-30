"""Build the domain knowledge graph for any use case.

The factory grounds generation in a knowledge graph. Four ways to get one,
chosen by ``domain.kg_source`` in the run config:

- ``tools`` — derived from MCP tool schemas alone. No LLM, no docs, no
  setup: every tool becomes an entity, every parameter becomes an entity,
  tools sharing a parameter name+type are linked. Enough to ground
  tool-centric questions ("which tools touch the customer record?").
- ``corpus`` — LLM-extracted from the enterprise's own docs (markdown /
  text in ``corpus_dir``). Each chunk is mined for entities + relations;
  results are merged, deduplicated, and capped. This is the path that
  makes the factory domain-aware for arbitrary businesses.
- ``spec`` — an explicit YAML hand-authored by the enterprise:
  ``{entities: [{name, type, description}], relations: [{head, relation,
  tail, description}]}``. Full control, zero inference.
- ``tables`` — measured from the enterprise's own data tables (CSV / JSON
  in ``tables_dir``) by the TabPFN probe stage
  (:mod:`bridge.tabpfn_probe`; see ``docs/adr-tabpfn-tabular-input.md``).
  A structural graph (tables, columns, measured ``predicts`` relations)
  is derived from the probe findings; when an LLM is available the
  findings are also narrated into documents and mined via the corpus
  path, and the two graphs are merged.

All three produce a :mod:`networkx` graph using GraphGen's
``light_rag_kg_builder`` attribute schema (``entity_name`` /
``entity_type`` / ``description`` on nodes; ``relation`` / ``description``
on edges), so :mod:`bridge.kg_context_exporter` consumes it directly.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import networkx as nx

from bridge.config import ConfigError


class KGBuildError(ConfigError):
    """The KG source could not be built (bad spec, bad corpus, bad source)."""


def _add_entity(
    g: nx.Graph,
    name: str,
    entity_type: str,
    description: str = "",
) -> str:
    key = name.strip()
    if not key:
        raise KGBuildError("entity with empty name")
    if g.has_node(key):
        node = g.nodes[key]
        # Merge: keep the longest description, union types.
        if len(description) > len(node.get("description", "")):
            node["description"] = description
        types = set(node.get("entity_type", "").split("+")) | {entity_type}
        node["entity_type"] = "+".join(sorted(t for t in types if t))
        return key
    g.add_node(
        key,
        entity_name=key,
        entity_type=entity_type,
        description=description or key,
    )
    return key


def _add_relation(
    g: nx.Graph, head: str, relation: str, tail: str, description: str = ""
) -> None:
    if head == tail or not relation.strip():
        return
    if g.has_edge(head, tail):
        return
    g.add_edge(
        head,
        tail,
        relation=relation.strip(),
        description=description or f"{head} {relation} {tail}",
    )


# ---------------------------------------------------------------------------
# tools source: from MCP tool descriptors, no LLM
# ---------------------------------------------------------------------------


def build_kg_from_tools(
    descriptors: Sequence[Any],
    max_entities: int = 200,
) -> nx.Graph:
    """Build a KG from tool descriptors (name/description/input_schema)."""
    g = nx.Graph()
    param_owners: Dict[tuple, List[str]] = {}
    for desc in descriptors:
        name = getattr(desc, "name", str(desc))
        tool_key = _add_entity(
            g, name, "tool", (getattr(desc, "description", "") or "")[:500]
        )
        schema = getattr(desc, "input_schema", {}) or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", []) or [])
        for prop_name, prop_spec in list(props.items())[:25]:  # cap per tool
            ptype = prop_spec.get("type", "string") if isinstance(prop_spec, dict) else "string"
            pdesc = prop_spec.get("description", "") if isinstance(prop_spec, dict) else ""
            param_key = _add_entity(g, f"{tool_key}.{prop_name}", "parameter", str(pdesc)[:300])
            _add_relation(
                g, tool_key, "has_parameter", param_key,
                f"{tool_key} takes parameter {prop_name} ({ptype})"
                + (" [required]" if prop_name in required else ""),
            )
            param_owners.setdefault((str(prop_name), str(ptype)), []).append(tool_key)
    # Link tools that speak about the same parameter: the composability
    # signal the sampler and the question generator both use.
    for (prop_name, ptype), owners in sorted(param_owners.items()):
        for a in owners:
            for b in owners:
                if a < b:
                    _add_relation(
                        g, a, "shares_parameter_with", b,
                        f"both take {prop_name} ({ptype})",
                    )
    return _cap_entities(g, max_entities)


def _cap_entities(g: nx.Graph, max_entities: int) -> nx.Graph:
    if g.number_of_nodes() <= max_entities:
        return g
    ranked = sorted(g.degree, key=lambda kv: (-kv[1], kv[0]))
    keep = {n for n, _ in ranked[:max_entities]}
    return g.subgraph(keep).copy()


# ---------------------------------------------------------------------------
# spec source: explicit enterprise YAML
# ---------------------------------------------------------------------------


def build_kg_from_spec(spec: Mapping[str, Any], max_entities: int = 200) -> nx.Graph:
    """Build a KG from an explicit ``{entities, relations}`` mapping."""
    if not isinstance(spec, Mapping):
        raise KGBuildError("spec must be a mapping with 'entities' and 'relations'")
    entities = spec.get("entities", []) or []
    relations = spec.get("relations", []) or []
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise KGBuildError("spec 'entities' and 'relations' must be lists")
    g = nx.Graph()
    for i, ent in enumerate(entities):
        if not isinstance(ent, dict) or not ent.get("name"):
            raise KGBuildError(f"spec entities[{i}]: needs a 'name'")
        _add_entity(
            g,
            str(ent["name"]),
            str(ent.get("type", "concept")),
            str(ent.get("description", ""))[:1000],
        )
    for i, rel in enumerate(relations):
        if not isinstance(rel, dict) or not all(
            rel.get(k) for k in ("head", "relation", "tail")
        ):
            raise KGBuildError(
                f"spec relations[{i}]: needs 'head', 'relation', 'tail'"
            )
        for endpoint in (rel["head"], rel["tail"]):
            if not g.has_node(str(endpoint)):
                _add_entity(g, str(endpoint), "concept")
        _add_relation(
            g,
            str(rel["head"]),
            str(rel["relation"]),
            str(rel["tail"]),
            str(rel.get("description", ""))[:500],
        )
    return _cap_entities(g, max_entities)


def load_spec_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise KGBuildError("pyyaml is required to load a YAML KG spec") from exc
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise KGBuildError(f"{path}: spec root must be a mapping")
    return spec


# ---------------------------------------------------------------------------
# corpus source: LLM extraction from enterprise docs
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
Extract the key entities and relationships from the document chunk below.
Reply with ONLY a JSON object of this shape, no prose, no code fences:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"head": "...", "relation": "...", "tail": "...", "description": "..."}}]}}
Keep names short and canonical (e.g. "customer record", not "the customer record mentioned above").
Limit to at most 15 entities and 20 relations.

--- chunk ---
{chunk}
"""


def _parse_extraction(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise KGBuildError(f"LLM extraction did not return JSON: {raw[:200]!r}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise KGBuildError(f"LLM extraction returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise KGBuildError("LLM extraction JSON must be an object")
    return parsed


def _chunk_text(text: str, chunk_chars: int = 4000) -> List[str]:
    chunks, current = [], []
    size = 0
    for para in text.split("\n\n"):
        if size + len(para) > chunk_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]


def _iter_corpus_files(corpus_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(corpus_dir):
        for fname in sorted(files):
            if fname.lower().endswith((".md", ".markdown", ".txt")):
                paths.append(os.path.join(root, fname))
    return paths


def build_kg_from_corpus(
    corpus_dir: str,
    llm_fn,
    max_entities: int = 200,
    max_chunks: int = 60,
) -> nx.Graph:
    """Build a KG by LLM extraction over the enterprise's docs."""
    if llm_fn is None:
        raise KGBuildError("kg_source='corpus' needs an LLM backend")
    if not os.path.isdir(corpus_dir):
        raise KGBuildError(f"corpus_dir not found: {corpus_dir}")
    files = _iter_corpus_files(corpus_dir)
    if not files:
        raise KGBuildError(f"no .md/.txt documents in {corpus_dir}")
    g = nx.Graph()
    chunks_done = 0
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for chunk in _chunk_text(text):
            if chunks_done >= max_chunks:
                break
            parsed = _parse_extraction(llm_fn(_EXTRACT_PROMPT.format(chunk=chunk[:4000])))
            for ent in parsed.get("entities", []) or []:
                if isinstance(ent, dict) and ent.get("name"):
                    _add_entity(
                        g,
                        str(ent["name"]),
                        str(ent.get("type", "concept")),
                        str(ent.get("description", ""))[:500],
                    )
            for rel in parsed.get("relations", []) or []:
                if (
                    isinstance(rel, dict)
                    and rel.get("head")
                    and rel.get("relation")
                    and rel.get("tail")
                ):
                    head = _add_entity(g, str(rel["head"]), "concept")
                    tail = _add_entity(g, str(rel["tail"]), "concept")
                    _add_relation(
                        g, head, str(rel["relation"]), tail,
                        str(rel.get("description", ""))[:300],
                    )
            chunks_done += 1
        if chunks_done >= max_chunks:
            break
    if g.number_of_nodes() == 0:
        raise KGBuildError("corpus extraction produced no entities")
    return _cap_entities(g, max_entities)


# ---------------------------------------------------------------------------
# tables source: TabPFN probe findings over enterprise data tables
# ---------------------------------------------------------------------------


def build_kg_from_tables(
    tables_dir: str,
    llm_fn=None,
    probe_factory=None,
    table_targets: Optional[Dict[str, str]] = None,
    work_dir: str = "",
    max_entities: int = 200,
) -> nx.Graph:
    """Build a KG from enterprise data tables via the TabPFN probe stage.

    Structural graph first (tables, columns, measured ``predicts``
    relations, straight from the findings — no LLM needed); when ``llm_fn``
    is available, the findings are narrated into a corpus directory and
    mined with the corpus extractor, and that graph is merged in.
    """
    from bridge import tabpfn_probe

    if not os.path.isdir(tables_dir):
        raise KGBuildError(f"tables_dir not found: {tables_dir}")
    tables = tabpfn_probe.load_tables_dir(tables_dir)
    if not tables:
        raise KGBuildError(f"no .csv/.json tables in {tables_dir}")
    findings = tabpfn_probe.probe_tables(
        tables, targets_by_table=table_targets, probe_factory=probe_factory
    )
    g = nx.Graph()
    for tf in findings["tables"]:
        tkey = _add_entity(
            g, tf["table"], "table",
            f"Enterprise data table with {tf['num_rows']} rows",
        )
        for col in tf["columns"]:
            desc = (
                f"Column of {tf['table']}: {col['type']}, "
                f"{col['cardinality']} distinct values"
            )
            ckey = _add_entity(g, f"{tf['table']}.{col['name']}", "column", desc)
            _add_relation(g, tkey, "has_column", ckey, desc)
        for tgt in tf["targets"]:
            metric = "accuracy" if tgt["task"] == "classification" else "R2"
            for imp in tgt["importances"][:5]:
                if imp["importance"] <= 0:
                    continue
                _add_relation(
                    g,
                    f"{tf['table']}.{imp['feature']}",
                    "predicts",
                    f"{tf['table']}.{tgt['column']}",
                    f"{imp['feature']} predicts {tgt['column']} "
                    f"(permutation importance {imp['importance']}, "
                    f"holdout {metric} {tgt['holdout_score']} vs "
                    f"baseline {tgt['baseline_score']})",
                )
    if work_dir:
        corpus_dir = work_dir
    else:
        import tempfile

        corpus_dir = tempfile.mkdtemp(prefix="tables_corpus_")
    tabpfn_probe.write_tables_corpus(findings, corpus_dir, llm_fn=llm_fn)
    if llm_fn is not None:
        extracted = build_kg_from_corpus(
            corpus_dir, llm_fn, max_entities=max_entities
        )
        for node, attrs in extracted.nodes(data=True):
            _add_entity(
                g,
                str(node),
                str(attrs.get("entity_type", "concept")),
                str(attrs.get("description", "")),
            )
        for head, tail, attrs in extracted.edges(data=True):
            _add_relation(
                g, str(head), str(attrs.get("relation", "related_to")),
                str(tail), str(attrs.get("description", "")),
            )
    return _cap_entities(g, max_entities)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def build_kg(
    source: str,
    *,
    descriptors: Optional[Sequence[Any]] = None,
    corpus_dir: str = "",
    spec: Optional[Mapping[str, Any]] = None,
    spec_file: str = "",
    llm_fn=None,
    max_entities: int = 200,
    tables_dir: str = "",
    probe_factory=None,
    table_targets: Optional[Dict[str, str]] = None,
    tables_work_dir: str = "",
) -> nx.Graph:
    """Build the domain KG from the configured source."""
    if source == "tools":
        return build_kg_from_tools(descriptors or [], max_entities=max_entities)
    if source == "spec":
        data = spec if spec is not None else (
            load_spec_file(spec_file) if spec_file else None
        )
        if data is None:
            raise KGBuildError("kg_source='spec' needs spec or spec_file")
        return build_kg_from_spec(data, max_entities=max_entities)
    if source == "corpus":
        if not corpus_dir:
            raise KGBuildError("kg_source='corpus' needs corpus_dir")
        return build_kg_from_corpus(
            corpus_dir, llm_fn, max_entities=max_entities
        )
    if source == "tables":
        if not tables_dir:
            raise KGBuildError("kg_source='tables' needs tables_dir")
        return build_kg_from_tables(
            tables_dir,
            llm_fn=llm_fn,
            probe_factory=probe_factory,
            table_targets=table_targets,
            work_dir=tables_work_dir,
            max_entities=max_entities,
        )
    raise KGBuildError(f"unknown kg_source: {source!r}")
