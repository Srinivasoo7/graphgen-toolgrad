"""Phase 4 — SFT dataset assembly.

Merges the bridge's tool-grounded QA pairs (Phase 3 ChatML) with GraphGen's
existing SFT pairs into one training mix:

    filter by quality -> dedupe -> shuffle -> train/valid split
      -> sft_mix.jsonl + dataset_card.md

GraphGen's SFT rows are ChatML JSONL: ``{"messages": [{"role", "content"}]}``
(``bases/base_generator.py::format_generation_results``). Bridge rows carry
an extra ``provenance`` object per example (KG entities, tool chain,
verification status) — a compatible superset, so the same fine-tuning
recipe consumes both. ShareGPT rows (``{"conversations": [...]}``) are
normalized to ChatML on the way in.
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bridge import trace_to_qa

_WS_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+", re.UNICODE)


class MixConfig:
    """Cutoffs and split policy for the SFT mix."""

    def __init__(
        self,
        valid_frac: float = 0.1,
        seed: int = 42,
        dedupe: bool = True,
        min_coverage: float = 0.0,
        require_chain_valid: bool = True,
        require_entities: bool = False,
    ) -> None:
        self.valid_frac = valid_frac
        self.seed = seed
        self.dedupe = dedupe
        self.min_coverage = min_coverage
        self.require_chain_valid = require_chain_valid
        self.require_entities = require_entities

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid_frac": self.valid_frac,
            "seed": self.seed,
            "dedupe": self.dedupe,
            "min_coverage": self.min_coverage,
            "require_chain_valid": self.require_chain_valid,
            "require_entities": self.require_entities,
        }


def normalize_question(question: str) -> str:
    """Canonical form for dedupe: lowercase, no punctuation, single spaces."""
    return _WS.sub(" ", _WS_PUNCT.sub("", question.lower())).strip()


def to_sft_record(qa: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge QA pair -> SFT row: ChatML messages + provenance metadata."""
    return {
        "messages": trace_to_qa.TraceToQAOperator.to_chatml(qa),
        "provenance": {
            "source": "graphgen_toolgrad_bridge",
            "origin": qa.get("provenance", {}).get("source"),
            "question": qa.get("question", ""),
            "entity_refs": qa.get("entity_refs", []),
            "required_tools": qa.get("required_tools", []),
            "chain": qa.get("chain", []),
            "verification": qa.get("verification", {}),
            "coverage": qa.get("coverage"),
            **{k: v for k, v in qa.get("provenance", {}).items()
               if k not in ("source", "question", "entity_refs",
                            "required_tools", "chain")},
        },
    }


def normalize_graphgen_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """GraphGen SFT row -> bridge SFT row (ChatML + tagged provenance).

    Accepts ChatML (``{"messages": [...]}``) and ShareGPT
    (``{"conversations": [{"from": "human"|"gpt", "value": ...}]}``) shapes.
    """
    if "messages" in rec:
        messages = rec["messages"]
    elif "conversations" in rec:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [
            {"role": role_map.get(t.get("from", ""), t.get("from", "")),
             "content": t.get("value", "")}
            for t in rec["conversations"]
        ]
    else:
        raise ValueError(f"unrecognized SFT row shape: {sorted(rec.keys())}")
    return {
        "messages": messages,
        "provenance": {
            "source": "graphgen",
            **{k: v for k, v in rec.items() if k not in ("messages", "conversations")},
        },
    }


def _passes_cutoffs(qa: Dict[str, Any], config: MixConfig) -> bool:
    verification = qa.get("verification") or {}
    if config.require_chain_valid and not verification.get("chain_valid", False):
        return False
    coverage = qa.get("coverage")
    if coverage is not None and coverage < config.min_coverage:
        return False
    if config.require_entities and not qa.get("entity_refs"):
        return False
    return True


def dedupe_pairs(pairs: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Drop pairs with duplicate normalized questions; returns (unique, n_dupes)."""
    seen = set()
    unique: List[Dict[str, Any]] = []
    for qa in pairs:
        key = normalize_question(qa.get("question", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(qa)
    return unique, len(pairs) - len(unique)


def assemble(
    pairs: Sequence[Dict[str, Any]],
    graphgen_pairs: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[MixConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build the SFT mix; returns (train, valid, report).

    ``pairs`` are bridge QA dicts (optionally pre-scored with
    ``verification``/``coverage`` keys, as emitted by the refinement loop).
    ``graphgen_pairs`` are raw GraphGen SFT rows. Quality cutoffs apply to
    bridge pairs only — GraphGen's own pairs are assumed already curated.
    """
    config = config or MixConfig()
    report: Dict[str, Any] = {
        "config": config.as_dict(),
        "n_bridge_in": len(pairs),
        "n_graphgen_in": len(graphgen_pairs or []),
    }

    kept = [qa for qa in pairs if _passes_cutoffs(qa, config)]
    report["n_bridge_after_cutoffs"] = len(kept)

    n_dupes = 0
    if config.dedupe:
        kept, n_dupes = dedupe_pairs(kept)
    report["n_bridge_dupes_removed"] = n_dupes

    records = [to_sft_record(qa) for qa in kept]
    records += [normalize_graphgen_record(rec) for rec in (graphgen_pairs or [])]
    report["n_records_total"] = len(records)

    rng = random.Random(config.seed)
    rng.shuffle(records)
    n_valid = max(1, int(len(records) * config.valid_frac)) if records else 0
    valid = records[:n_valid]
    train = records[n_valid:]
    report["n_train"] = len(train)
    report["n_valid"] = len(valid)
    report["sources"] = {
        source: sum(1 for r in records if r["provenance"].get("source") == source)
        for source in {r["provenance"].get("source") for r in records}
    }
    return train, valid, report


def write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> str:
    """Write SFT rows as JSONL; returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def write_dataset_card(
    report: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]],
    path: str,
) -> str:
    """Write a markdown dataset card documenting the mix; returns the path."""
    lines = [
        "# SFT dataset card — graphgen-toolgrad bridge",
        "",
        "Knowledge-grounded, tool-use-verified synthetic SFT data: every bridge",
        "example names real KG entities and its answer is backed by an executed",
        "tool chain (verified, not imagined).",
        "",
        "## Composition",
        "",
        f"- Bridge pairs in: {report['n_bridge_in']}",
        f"- After quality cutoffs: {report['n_bridge_after_cutoffs']}",
        f"- Duplicates removed: {report['n_bridge_dupes_removed']}",
        f"- GraphGen pairs merged: {report['n_graphgen_in']}",
        f"- Total records: {report['n_records_total']} "
        f"(train {report['n_train']} / valid {report['n_valid']})",
        f"- Sources: {json.dumps(report['sources'])}",
        "",
        "## Quality gates (bridge pairs)",
        "",
        f"- require_chain_valid: {report['config']['require_chain_valid']}",
        f"- min_coverage: {report['config']['min_coverage']}",
        f"- require_entities: {report['config']['require_entities']}",
        "",
        "## Eval snapshot",
        "",
    ]
    if eval_metrics:
        lines += [
            f"- Chain-verified fraction: {eval_metrics.get('fraction_chain_verified')}",
            f"- Mean ToolKG coverage: {eval_metrics.get('mean_toolkg_coverage')}",
            f"- Entity-grounding rate: {eval_metrics.get('entity_grounding_rate')}",
            f"- Parametric-answerability flags: "
            f"{eval_metrics.get('parametric_answerability', {}).get('flagged')}",
        ]
    else:
        lines.append("- (no eval harness run recorded)")
    lines += [
        "",
        "## Limits",
        "",
        "- Bridge quality gates are necessary, not sufficient: a verified chain",
        "  and grounded entities do not guarantee a good training example.",
        "- GraphGen-merged pairs carry their own provenance; they were not",
        "  re-verified by this pipeline.",
        "- Mechanical (template) drafts without an LLM pass are plumbing",
        "  fixtures, not training data — see docs/phase3-trace2qa.md.",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
