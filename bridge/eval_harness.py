"""Phase 4 — keyless evaluation harness.

Dataset-level metrics for a generated SFT mix, WITHOUT a trained model.
This harness cannot tell you whether fine-tuning on the data helps — that
needs the live SFT run (see docs/phase4-unified-loop.md). What it can tell
you, cheaply and deterministically:

- ``fraction_chain_verified``: share of pairs whose tool chain actually
  executes (needs an ``executor`` or precomputed ``verifications``).
- ``mean_toolkg_coverage``: chain <-> ToolKG agreement (Phase 3 feature).
- ``entity_grounding_rate``: share of pairs anchored to real KG entities.
- ``tool_diversity``: distinct tools, step counts, most-used tools.
- ``length_stats``: token-count distributions for questions/answers.
- ``parametric_answerability``: flags pairs whose answer appears verbatim
  in the KG context text — those may be answerable from parametric/domain
  knowledge without running any tools, which weakens the "tool-grounded"
  claim. This is a heuristic flag, not a proof: a flagged pair is
  *suspect*, an unflagged pair is not *proven* tool-dependent.
"""

import re
from collections import Counter
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

from bridge import chain_verifier
from bridge.kg_context_exporter import estimate_tokens, render_kg_context

_WS = re.compile(r"\s+", re.UNICODE)


def _canon(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def _stats(values: Sequence[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0, "n": 0}
    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def parametric_answerability_flags(
    pairs: Sequence[Dict[str, Any]],
    kg_context: Optional[Dict[str, Any]],
    min_answer_chars: int = 40,
) -> Dict[str, Any]:
    """Flag pairs whose answer draft is verbatim in the KG context text.

    Returns ``{"flagged": int, "fraction": float, "flagged_indices": [...]}``.
    Empty KG context -> nothing flaggable (0 flags, not an error).
    """
    kg_text = _canon(render_kg_context(kg_context)) if kg_context else ""
    flagged_indices = []
    for i, qa in enumerate(pairs):
        answer = _canon(qa.get("answer_draft", ""))
        if len(answer) >= min_answer_chars and kg_text and answer in kg_text:
            flagged_indices.append(i)
    n = len(pairs)
    return {
        "flagged": len(flagged_indices),
        "fraction": (len(flagged_indices) / n) if n else 0.0,
        "flagged_indices": flagged_indices,
    }


def evaluate(
    pairs: Sequence[Dict[str, Any]],
    kg_context: Optional[Dict[str, Any]] = None,
    toolkg: Any = None,
    executor: Any = None,
    verifications: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute dataset-level metrics for ``pairs`` (bridge QA dicts).

    Chain verification needs either ``executor`` (runs ``verify_qa`` per
    pair) or ``verifications`` (precomputed ``verify_qa`` outputs, e.g. from
    the refinement loop). With neither, ``fraction_chain_verified`` is None
    and the report says so — the harness refuses to guess.
    """
    pairs = list(pairs)
    n = len(pairs)
    metrics: Dict[str, Any] = {"n": n}

    # -- chain verification -------------------------------------------
    if verifications is not None:
        if len(verifications) != n:
            raise ValueError(
                f"verifications has {len(verifications)} entries for {n} pairs"
            )
        valid = [bool(v.get("chain_valid")) for v in verifications]
    elif executor is not None:
        valid = [bool(chain_verifier.verify_qa(qa, executor)["chain_valid"]) for qa in pairs]
    else:
        valid = None
    metrics["fraction_chain_verified"] = (sum(valid) / n) if valid is not None and n else (
        None if valid is None else 0.0
    )
    if valid is None:
        metrics["verification_note"] = (
            "no executor or precomputed verifications supplied; "
            "chain validity not measured"
        )

    # -- ToolKG coverage ----------------------------------------------
    coverages = [
        chain_verifier.chain_toolkg_coverage(qa.get("chain", []), toolkg)
        for qa in pairs
    ] if toolkg is not None else []
    metrics["mean_toolkg_coverage"] = mean(coverages) if coverages else None
    if toolkg is None:
        metrics["coverage_note"] = "no ToolKG supplied; coverage not measured"

    # -- entity grounding ----------------------------------------------
    entity_names = (
        {e.get("name") for e in kg_context.get("entities", []) if e.get("name")}
        if kg_context else set()
    )
    grounded = 0
    for qa in pairs:
        refs = qa.get("entity_refs", []) or []
        question_lower = qa.get("question", "").lower()
        if (refs and all(r in entity_names for r in refs)
                and any(r.lower() in question_lower for r in refs)):
            grounded += 1
    metrics["entity_grounding_rate"] = (grounded / n) if n else 0.0

    # -- tool diversity --------------------------------------------------
    tool_counter: Counter = Counter()
    total_steps = 0
    for qa in pairs:
        tools = qa.get("required_tools", []) or []
        tool_counter.update(tools)
        total_steps += len(tools)
    metrics["tool_diversity"] = {
        "distinct_tools": len(tool_counter),
        "total_steps": total_steps,
        "mean_steps_per_pair": (total_steps / n) if n else 0.0,
        "most_common": tool_counter.most_common(5),
    }

    # -- length stats ----------------------------------------------------
    metrics["length_stats"] = {
        "question_tokens": _stats([estimate_tokens(qa.get("question", "")) for qa in pairs]),
        "answer_tokens": _stats([estimate_tokens(qa.get("answer_draft", "")) for qa in pairs]),
    }

    # -- parametric answerability ----------------------------------------
    metrics["parametric_answerability"] = parametric_answerability_flags(pairs, kg_context)

    # -- honest limits -----------------------------------------------------
    metrics["limits"] = [
        "Dataset-level only: says nothing about whether SFT on this data "
        "improves a model. That needs the live fine-tune + eval.",
        "parametric_answerability is a verbatim-overlap heuristic: flagged "
        "pairs are suspect, unflagged pairs are not proven tool-dependent.",
        "Chain verification replays recorded inputs; it does not check that "
        "the answer draft faithfully summarizes the results.",
    ]
    return metrics
