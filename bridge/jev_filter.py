"""Jev-backed quality filter (experiment).

Drop-in alternative to :func:`refinement_loop.filter_pairs` that asks Jev —
TypeSafe's System One decision model — for an accept/reject verdict on each
scored QA pair instead of applying the keyless heuristic gates.

Jev answers one ``choice`` question (accept vs reject) and one ``noul``
question (is the question grounded in a KG entity?) over a compact rendering
of the pair plus its keyless scores. The verdict carries a calibrated
probability, so a confidence threshold decides. Anything Jev cannot judge
(transport error, malformed response) falls back to the heuristic gates,
never to a blind accept.

The bridge never touches the API key: calls go through the
``typesafe`` skill's ``jev_decide`` CLI, which authenticates via the stored
credential.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from bridge.refinement_loop import RefineConfig, passes_thresholds

DEFAULT_MODEL = "jev-latest"
DEFAULT_CLI = os.path.expanduser(
    os.environ.get(
        "JEV_CLI", "~/workspace/skills/typesafe/bin/jev_decide.py"
    )
)
MIN_CONFIDENCE = 0.5


def _compact_chain(qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "tool": step.get("tool"),
            "ok": bool((step.get("result_preview") or "") != "ERROR"),
        }
        for step in (qa.get("chain") or [])
    ]


def build_request(
    qa: Dict[str, Any], score: Dict[str, Any], model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Build the Jev ``/v1/systemone`` request document for one scored pair."""
    entity_refs = score.get("entity_refs") or []
    return {
        "model": model,
        "state": {
            "question": qa.get("question", ""),
            "answer_draft": qa.get("answer_draft", ""),
            "chain": _compact_chain(qa),
            "keyless_scores": {
                "chain_valid": bool(score.get("chain_valid")),
                "coverage": float(score.get("coverage") or 0.0),
                "entity_grounded": bool(score.get("entity_grounded")),
                "entity_refs": entity_refs,
            },
        },
        "questions": {
            "verdict": {
                "type": "choice",
                "instructions": (
                    "Should this QA pair be accepted into an SFT training "
                    "dataset for a tool-using agent? Accept only if the tool "
                    "chain is valid and executable, the question is answered "
                    "by executing the chain, and the question names a real "
                    "entity from the knowledge-graph context."
                ),
                "criteria": {
                    "accept": (
                        "Good training data: valid chain, composable tools, "
                        "question grounded in a KG entity, answer supported "
                        "by the chain."
                    ),
                    "reject": (
                        "Defective: broken chain, uncomposable tools, "
                        "ungrounded question, or an answer the chain does "
                        "not support."
                    ),
                },
            },
            "grounded": {
                "type": "noul",
                "instructions": (
                    "Does the question name a specific real-world entity — a "
                    "proper noun such as a product, company, or place — "
                    "instead of referring to it generically ('the factory', "
                    "'the records', 'it')?"
                ),
            },
        },
    }


def _parse_decision(
    response: Dict[str, Any], min_confidence: float
) -> Tuple[str, float, Optional[float]]:
    """Extract (verdict, p_accept, grounded) from a Jev response.

    Uses the ``probabilities`` map (P(accept)) rather than the opaque
    top-level ``confidence``, which is a margin, not a probability.
    Raises ValueError on anything unusable so the caller can fall back.
    """
    answers = response.get("answers") or {}
    verdict_ans = answers.get("verdict") or {}
    probs = verdict_ans.get("probabilities") or {}
    try:
        p_accept = float(probs["accept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"no usable P(accept): {probs!r}") from exc
    if not 0.0 <= p_accept <= 1.0:
        raise ValueError(f"P(accept) out of range: {p_accept!r}")
    grounded_ans = answers.get("grounded") or {}
    grounded = grounded_ans.get("noul")
    grounded = float(grounded) if grounded is not None else None
    verdict = "accept" if p_accept >= min_confidence else "reject"
    return verdict, p_accept, grounded


class JevDecider:
    """Ask Jev for accept/reject verdicts via the skill CLI (subprocess).

    ``decide`` never raises on provider trouble: failures fall back to the
    heuristic gates and are recorded in the returned dict.
    """

    def __init__(
        self,
        cli: str = DEFAULT_CLI,
        model: str = DEFAULT_MODEL,
        min_confidence: float = MIN_CONFIDENCE,
        timeout: float = 120.0,
    ) -> None:
        self.cli = cli
        self.model = model
        self.min_confidence = min_confidence
        self.timeout = timeout

    def decide(
        self,
        qa: Dict[str, Any],
        score: Dict[str, Any],
        config: Optional[RefineConfig] = None,
    ) -> Dict[str, Any]:
        request_doc = build_request(qa, score, self.model)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, self.cli],
                input=json.dumps(request_doc),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            latency_ms = (time.monotonic() - started) * 1000.0
            if proc.returncode != 0:
                raise RuntimeError(f"jev_decide exited {proc.returncode}: {proc.stdout[:300]}")
            response = json.loads(proc.stdout)
            if response.get("error"):
                raise RuntimeError(f"provider error: {response.get('error')}: {response.get('detail', '')[:300]}")
            verdict, confidence, grounded = _parse_decision(response, self.min_confidence)
            usage = response.get("usage") or {}
            return {
                "verdict": verdict,
                "confidence": confidence,
                "grounded": grounded,
                "usage": usage,
                "latency_ms": round(latency_ms, 1),
                "model": response.get("model", self.model),
                "source": "jev",
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - fallback must never crash the loop
            latency_ms = (time.monotonic() - started) * 1000.0
            cfg = config or RefineConfig()
            heuristic = "accept" if passes_thresholds(score, cfg) else "reject"
            return {
                "verdict": heuristic,
                "confidence": 0.0,
                "grounded": None,
                "usage": {},
                "latency_ms": round(latency_ms, 1),
                "model": self.model,
                "source": "heuristic_fallback",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }


def jev_filter_pairs(
    scored: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    config: RefineConfig,
    decider: JevDecider,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]]]:
    """Split scored ``(qa, score)`` pairs into (kept, rejected) via Jev.

    Same return contract as :func:`refinement_loop.filter_pairs`; kept pairs
    carry a small ``qa["jev"]`` annotation (verdict, confidence, latency,
    model, source) for the dataset card.
    """
    kept: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    rejected: List[Dict[str, Any]] = []
    for qa, score in scored:
        decision = decider.decide(qa, score, config)
        qa = dict(qa)
        qa["jev"] = {
            "verdict": decision["verdict"],
            "confidence": decision["confidence"],
            "grounded": decision["grounded"],
            "latency_ms": decision["latency_ms"],
            "model": decision["model"],
            "source": decision["source"],
            "input_tokens": (decision["usage"] or {}).get("input_tokens"),
            "error": decision["error"],
        }
        if decision["verdict"] == "accept":
            kept.append((qa, score))
        else:
            rejected.append(qa)
    return kept, rejected
