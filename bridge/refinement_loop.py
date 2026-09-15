"""Phase 4 — unified refinement loop.

Orchestrates the full bridge pipeline end to end:

    KG context (Phase 1) -> ToolKG-guided sampling (Phase 2)
      -> ToolGrad generation -> trace-to-QA (Phase 3)
      -> chain verification -> quality filtering
      -> refinement iterations (ToolGrad's textual-gradient pattern)

Every LLM call goes through an injected ``llm_fn(prompt) -> str`` (stub in
tests, real backend in the live run). With ``llm_fn=None`` the loop runs
generation + filtering once and stops — refinement genuinely needs an LLM,
and the ledger records why.

Per-iteration metrics are recorded in a run ledger (JSON-serializable dict)
via :func:`save_ledger` / :func:`load_ledger`.
"""

import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bridge import chain_verifier, trace_to_qa
from bridge.kg_context_exporter import estimate_tokens


class RefineConfig:
    """Thresholds and budgets for the refinement loop."""

    def __init__(
        self,
        max_iterations: int = 3,
        min_coverage: float = 0.5,
        require_chain_valid: bool = True,
        require_entities: bool = True,
        seed: int = 42,
    ) -> None:
        self.max_iterations = max(1, max_iterations)
        self.min_coverage = min_coverage
        self.require_chain_valid = require_chain_valid
        self.require_entities = require_entities
        self.seed = seed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "min_coverage": self.min_coverage,
            "require_chain_valid": self.require_chain_valid,
            "require_entities": self.require_entities,
            "seed": self.seed,
        }


def score_qa(
    qa: Dict[str, Any],
    executor: Any,
    toolkg: Any,
    kg_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Score one QA pair: verification, ToolKG coverage, entity grounding.

    Returns a flat score dict; ``passes`` is computed separately by
    :func:`filter_pairs` so tests can inspect raw scores.
    """
    verification = chain_verifier.verify_qa(qa, executor)
    coverage = chain_verifier.chain_toolkg_coverage(qa.get("chain", []), toolkg)
    entity_names = {
        e.get("name") for e in kg_context.get("entities", []) if e.get("name")
    }
    refs = qa.get("entity_refs", []) or []
    question_lower = qa.get("question", "").lower()
    entity_grounded = (
        bool(refs)
        and all(r in entity_names for r in refs)
        and any(r.lower() in question_lower for r in refs)
    )
    return {
        "chain_valid": verification["chain_valid"],
        "verification": verification,
        "coverage": coverage,
        "entity_grounded": entity_grounded,
        "entity_refs": refs,
        "required_tools": qa.get("required_tools", []),
        "question_tokens": estimate_tokens(qa.get("question", "")),
        "answer_tokens": estimate_tokens(qa.get("answer_draft", "")),
    }


def critique_qa(
    qa: Dict[str, Any],
    score: Dict[str, Any],
    config: RefineConfig,
) -> List[str]:
    """Rule-based textual critique of a QA pair (the textual-gradient analog).

    ToolGrad's inverse predictor improves query/response pairs by reacting
    to textual feedback about what's wrong; here the critic is deterministic
    and keyless: verification failures, low ToolKG coverage, and missing
    entity anchors each become an actionable critique line. The refinement
    prompt carries these to the LLM.
    """
    issues: List[str] = []
    verification = score["verification"]
    if not verification["chain_valid"]:
        for tool in verification.get("missing_tools", []):
            issues.append(
                f"The chain calls tool {tool!r} which is not available. "
                "Rewrite the question so it only needs available tools, "
                "or fix the tool name."
            )
        for mismatch in verification.get("schema_mismatches", []):
            issues.append(
                f"Step {mismatch['step']} ({mismatch['tool']}) has invalid "
                f"arguments: {'; '.join(mismatch['problems'])}. "
                "Correct the tool arguments."
            )
        for step in verification.get("executed_steps", []):
            if not step["ok"] and step["error"] and "missing tool" not in step["error"]:
                issues.append(
                    f"Step {step['step']} ({step['tool']}) failed when "
                    f"executed: {step['error']}. Adjust the chain usage."
                )
    if score["coverage"] < config.min_coverage:
        issues.append(
            f"Only {score['coverage']:.0%} of consecutive tool pairs are "
            "composable according to the ToolKG (minimum "
            f"{config.min_coverage:.0%}). Prefer tool sequences whose "
            "outputs feed the next tool's inputs."
        )
    if config.require_entities and not score["entity_grounded"]:
        issues.append(
            "The question is not anchored to any knowledge-graph entity. "
            "Name a real entity from the KG context in the question."
        )
    if not qa.get("answer_draft", "").strip():
        issues.append("The answer draft is empty. Write an answer that the tool chain results support.")
    return issues


def passes_thresholds(score: Dict[str, Any], config: RefineConfig) -> bool:
    """True iff a scored QA pair clears every configured quality gate."""
    if config.require_chain_valid and not score["chain_valid"]:
        return False
    if score["coverage"] < config.min_coverage:
        return False
    if config.require_entities and not score["entity_grounded"]:
        return False
    return True


def filter_pairs(
    scored: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    config: RefineConfig,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]]]:
    """Split scored ``(qa, score)`` pairs into (kept, rejected).

    ``kept`` retains the ``(qa, score)`` tuples so callers can attach
    verification/coverage to the surviving records; ``rejected`` is plain
    QA dicts (they go back for refinement or are dropped).
    """
    kept = [(qa, score) for qa, score in scored if passes_thresholds(score, config)]
    rejected = [qa for qa, score in scored if not passes_thresholds(score, config)]
    return kept, rejected


def _attach_scores(kept: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Attach verification + coverage to kept QA records (for the SFT mix)."""
    out = []
    for qa, score in kept:
        qa = dict(qa)
        qa["verification"] = score["verification"]
        qa["coverage"] = score["coverage"]
        out.append(qa)
    return out


def _critique_block(issues: List[str]) -> str:
    lines = ["\n\n## Critique of your previous attempt",
             "Address EVERY point below in your regenerated QUESTION and ANSWER:"]
    lines += [f"- {issue}" for issue in issues]
    return "\n".join(lines) + "\n"


class RefinementLoop:
    """Run generation -> verify -> filter -> refine until quality gates pass.

    ``executor`` verifies chains (``MockToolExecutor`` keyless,
    ``ToolGradExecutor`` live). ``samples`` are ToolGrad workflow-sample
    dicts; ``tracers`` optionally supply provenance (same contract as
    ``TraceToQAOperator.process``).
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        kg_context: Optional[Dict[str, Any]] = None,
        toolkg: Any = None,
        executor: Any = None,
        config: Optional[RefineConfig] = None,
    ) -> None:
        self.llm_fn = llm_fn
        self.kg_context = kg_context or {"entities": [], "triples": [], "communities": []}
        self.toolkg = toolkg
        self.executor = executor
        self.config = config or RefineConfig()
        self.rng = random.Random(self.config.seed)

    def _score_all(self, pairs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        return [(qa, score_qa(qa, self.executor, self.toolkg, self.kg_context)) for qa in pairs]

    @staticmethod
    def _iteration_metrics(
        iteration: int, scored: List[Tuple[Dict[str, Any], Dict[str, Any]]], num_refined: int
    ) -> Dict[str, Any]:
        scores = [s for _, s in scored]
        n = len(scores)
        return {
            "iteration": iteration,
            "num_pairs": n,
            "num_chain_valid": sum(1 for s in scores if s["chain_valid"]),
            "num_entity_grounded": sum(1 for s in scores if s["entity_grounded"]),
            "mean_coverage": (sum(s["coverage"] for s in scores) / n) if n else 0.0,
            "num_refined": num_refined,
        }

    def run(
        self,
        samples: Sequence[Dict[str, Any]],
        tracers: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute the loop; returns (kept_pairs, ledger)."""
        if self.executor is None:
            raise ValueError("RefinementLoop needs an executor to verify chains")
        op = trace_to_qa.TraceToQAOperator(
            llm_fn=self.llm_fn,
            kg_context=self.kg_context,
            toolkg=self.toolkg,
        )
        pairs, gen_stats = op.process(samples, tracers)
        # map each pair back to its chain for refinement prompts
        chain_by_id = {c["chain_id"]: c for s in samples for c in trace_to_qa.extract_chains(s)}

        ledger: Dict[str, Any] = {
            "config": self.config.as_dict(),
            "num_samples": len(samples),
            "gen_stats": gen_stats,
            "iterations": [],
            "stopped_reason": "",
        }

        current = pairs
        attempt: Dict[int, int] = {}
        for iteration in range(self.config.max_iterations):
            scored = self._score_all(current)
            kept, rejected = filter_pairs(scored, self.config)
            metrics = self._iteration_metrics(iteration, scored, num_refined=0)
            metrics["num_kept"] = len(kept)
            metrics["num_rejected"] = len(rejected)

            if not rejected:
                metrics["num_refined"] = 0
                ledger["iterations"].append(metrics)
                ledger["stopped_reason"] = "all_pass"
                return _attach_scores(kept), ledger

            if self.llm_fn is None:
                # Refinement is an LLM operation; without a backend there is
                # nothing honest to iterate on.
                ledger["iterations"].append(metrics)
                ledger["stopped_reason"] = "no_llm_refinement"
                return _attach_scores(kept), ledger

            # Refine the rejected pairs with their critiques.
            refined: List[Dict[str, Any]] = []
            for qa in rejected:
                score = next(s for q, s in scored if q is qa)
                issues = critique_qa(qa, score, self.config)
                chain = chain_by_id.get(qa["provenance"].get("chain_id"))
                if chain is None:
                    continue  # cannot regenerate without the chain; drop it
                attempt[id(qa)] = attempt.get(id(qa), 0) + 1
                new_pairs = op.generate_for_chain(
                    chain,
                    self.kg_context,
                    trace_seed=qa["provenance"].get("trace_seed"),
                    prompt_suffix=_critique_block(issues),
                )
                new_qa = new_pairs[0]
                new_qa["provenance"]["refinement_attempt"] = attempt[id(qa)]
                new_qa["provenance"]["critique"] = issues
                refined.append(new_qa)
            metrics["num_refined"] = len(refined)
            ledger["iterations"].append(metrics)
            current = [qa for qa, _ in kept] + refined

        # max_iterations exhausted: final filter on the last round
        scored = self._score_all(current)
        kept, _ = filter_pairs(scored, self.config)
        ledger["stopped_reason"] = "max_iterations"
        return _attach_scores(kept), ledger


def save_ledger(ledger: Dict[str, Any], path: str) -> str:
    """Write the run ledger as JSON; returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    return path


def load_ledger(path: str) -> Dict[str, Any]:
    """Read a run ledger back from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
