"""Phase 3: convert ToolGrad execution traces into tool-grounded QA pairs.

A ToolGrad run produces *verified* tool-use chains (queries + executed
``intermediate_steps`` + responses), but its queries are generic — Phase 1
grounds them in a domain KG. This module closes the loop the other way: it
consumes ToolGrad's execution traces and emits agentic QA pairs whose
questions name real KG entities and are answerable **only** by running the
recorded tool chain (not from the model's parametric knowledge).

Mirrors GraphGen's operator conventions:
- an operator class with ``process(batch) -> (results, stats)`` (cf.
  ``QuizService`` / ``GenerateService`` in GraphGen);
- the LLM is a swappable hook: GraphGen operators resolve a
  ``BaseLLMWrapper`` via ``init_llm("synthesizer")``; here the constructor
  accepts ``llm_fn(prompt) -> str``. Tests inject a stub (or ``None`` for
  the deterministic template generator); the live run passes a real
  backend. No LLM key is required to build, test, or run this module.

Input contracts (both from real ToolGrad code, no LLM needed):
- workflow sample: ``ApiUseWorkflow.model_dump()`` JSON — the per-run file
  ``examples/outputs/seed={seed}__iter={n}__num_apis={k}.json``
  (``{"query", "api_use_chains", "response"}``). This is the primary input:
  it carries the executed ``intermediate_steps`` per chain.
- tracer file (optional provenance): ``ExecutionTracer.save()`` JSON —
  ``{"seed", "total_iterations", "iterations": [...]}`` with per-iteration
  ``proposals`` / ``executions`` / ``selection`` / ``workflow_update``.

Output: one QA record per (chain, sample)::

    {
      "question": str,          # names real KG entities; needs the chain
      "answer_draft": str,      # drafted from tool results, not the response
      "required_tools": [str],  # tool names in chain order
      "chain": [                # normalized executed steps
        {"tool": str, "tool_input": dict, "result_preview": str}],
      "entity_refs": [str],     # KG entities the question is anchored to
      "provenance": {
        "source": "toolgrad_trace",
        "chain_id": str,
        "trace_seed": int | None,
        "toolkg_edges_used": int,
        "kg_entity_count": int,
      },
    }

``to_chatml`` renders a QA record in GraphGen's SFT format (ChatML), so the
pairs slot directly into the Phase 4 data mix.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bridge.kg_context_exporter import render_kg_context

_QA_PROMPT = """\
You are writing a training question for a tool-using agent. The agent can
only answer by executing the tool chain below; it cannot rely on general
knowledge.

Domain context (entities and relationships are REAL — use their exact names):
{kg_context}

Executed tool chain:
{chain_text}

Write ONE question that:
1. names one or more of the real entities above (use exact names),
2. can ONLY be answered by executing the tool chain (not from memory),
3. has its answer contained in the tool results shown.

Output format, exactly:
QUESTION: <the question>
ANSWER: <a draft answer citing the tool results>
"""

_ANSWER_FALLBACK = "(drafted mechanically from tool results; LLM refinement pending)"


def _preview(result: Any, max_chars: int = 300) -> str:
    """Best-effort string preview of a tool result (dict or str)."""
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            text = str(result)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def load_workflow_sample(path: str) -> dict:
    """Load a ToolGrad workflow sample (``ApiUseWorkflow.model_dump()`` JSON)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tracer(path: str) -> dict:
    """Load a ToolGrad ``ExecutionTracer.save()`` JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_chains(sample: dict) -> List[dict]:
    """Normalize ``sample["api_use_chains"]`` into step records.

    Returns ``[{"chain_id", "description", "steps": [{"tool", "tool_input",
    "result_preview"}]}]``. Skips chains with no executed steps.
    """
    records = []
    for chain_id, chain in (sample.get("api_use_chains") or {}).items():
        steps = []
        for item in chain.get("intermediate_steps") or []:
            action, result = item[0], item[1]
            tool = action.get("tool") if isinstance(action, dict) else getattr(action, "tool", None)
            tool_input = (
                action.get("tool_input", {})
                if isinstance(action, dict)
                else getattr(action, "tool_input", {})
            )
            if not tool:
                continue
            steps.append({
                "tool": str(tool),
                "tool_input": tool_input if isinstance(tool_input, dict) else {},
                "result_preview": _preview(result),
            })
        if steps:
            records.append({
                "chain_id": chain_id,
                "description": chain.get("description", ""),
                "steps": steps,
            })
    return records


def _entity_names(kg_context: dict) -> List[str]:
    return [e.get("name") for e in kg_context.get("entities", []) if e.get("name")]


def ground_entities(chain: dict, kg_context: dict) -> List[str]:
    """Deterministically anchor a chain to KG entities.

    An entity counts as referenced if its name (case-insensitive) appears
    in any tool input value or tool result preview of the chain. No LLM.
    """
    names = _entity_names(kg_context)
    if not names:
        return []
    haystack_parts = []
    for step in chain["steps"]:
        haystack_parts.append(step.get("result_preview", ""))
        for value in step.get("tool_input", {}).values():
            haystack_parts.append(str(value))
    haystack = "\n".join(haystack_parts).lower()
    return [n for n in names if n.lower() in haystack]


def _chain_text(chain: dict) -> str:
    lines = [f"Chain {chain['chain_id']}: {chain.get('description', '')}"]
    for i, step in enumerate(chain["steps"], 1):
        inputs = ", ".join(f"{k}={v!r}" for k, v in step["tool_input"].items())
        lines.append(f"  {i}. {step['tool']}({inputs}) -> {step['result_preview']}")
    return "\n".join(lines)


def _tool_summaries(toolkg) -> str:
    """One-line-per-tool summary from a Phase 2 ToolKG (may be None)."""
    if toolkg is None:
        return "(no tool catalog graph)"
    lines = []
    for node in sorted(toolkg.nodes):
        attrs = toolkg.nodes[node]
        desc = attrs.get("description", "")
        desc = desc if len(desc) <= 120 else desc[:120] + "..."
        lines.append(f"- {attrs.get('entity_name', node)}: {desc}")
    return "\n".join(lines) or "(empty tool catalog)"


def _template_generate(prompt_ctx: dict) -> Tuple[str, str]:
    """Deterministic no-LLM QA draft (used when ``llm_fn`` is None).

    Grounding comes from ``entity_refs`` (exact KG names matched against
    the chain) and the tool names — mechanical but honest, and it keeps
    the whole pipeline keyless.
    """
    chain = prompt_ctx["chain"]
    entity_refs = prompt_ctx["entity_refs"]
    tools = [s["tool"] for s in chain["steps"]]
    first = chain["steps"][0]
    anchor = ", ".join(entity_refs) if entity_refs else "the data"
    question = (
        f"Using the {', '.join(tools)} tools in order, what do the local "
        f"records say about {anchor}?"
    )
    answer = (
        f"{_ANSWER_FALLBACK} "
        f"Chain: {chain['chain_id']} "
        f"({chain.get('description', 'no description')}). "
        f"Key result: {first['result_preview']}"
    )
    return question, answer


class TraceToQAOperator:
    """GraphGen-style operator: ToolGrad traces -> tool-grounded QA pairs.

    ``process(samples)`` mirrors GraphGen operators' ``process(batch) ->
    (results, stats)`` contract. Each sample is a workflow-sample dict
    (``ApiUseWorkflow.model_dump()`` JSON); an optional ``tracer`` dict
    adds provenance.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        kg_context: Optional[dict] = None,
        max_pairs_per_chain: int = 1,
        toolkg=None,
        working_dir: str = "cache",
        op_name: str = "trace_to_qa",
    ) -> None:
        self.llm_fn = llm_fn
        self.kg_context = kg_context or {"entities": [], "triples": [], "communities": []}
        self.max_pairs_per_chain = max(1, max_pairs_per_chain)
        self.toolkg = toolkg
        self.working_dir = working_dir
        self.op_name = op_name

    # -- prompt construction / parsing (the LLM seam) --------------------

    def build_prompt(self, chain: dict, kg_context: dict) -> Tuple[str, dict]:
        """Build the QA-generation prompt; returns (prompt, prompt_ctx)."""
        entity_refs = ground_entities(chain, kg_context)
        prompt_ctx = {
            "chain": chain,
            "entity_refs": entity_refs,
            "kg_text": render_kg_context(kg_context),
            "tool_text": _tool_summaries(self.toolkg),
        }
        prompt = _QA_PROMPT.format(
            kg_context=prompt_ctx["kg_text"],
            chain_text=_chain_text(chain) + "\n\nAvailable tools:\n" + prompt_ctx["tool_text"],
        )
        return prompt, prompt_ctx

    @staticmethod
    def parse_llm_output(text: str) -> Dict[str, str]:
        """Parse ``QUESTION: ... / ANSWER: ...`` output; degrades gracefully."""
        question, answer = "", ""
        for line in text.splitlines():
            upper = line.strip().upper()
            if upper.startswith("QUESTION:"):
                question = line.split(":", 1)[1].strip()
            elif upper.startswith("ANSWER:"):
                answer = line.split(":", 1)[1].strip()
            elif question and not answer:
                question += " " + line.strip()
            elif answer:
                answer += " " + line.strip()
        if not question:
            question = text.strip().splitlines()[0].strip() if text.strip() else ""
        return {"question": question, "answer": answer or _ANSWER_FALLBACK}

    # -- main entry point ------------------------------------------------

    def generate_for_chain(
        self, chain: dict, kg_context: dict, trace_seed: Optional[int] = None
    ) -> List[dict]:
        """Emit QA pairs for one executed chain."""
        prompt, prompt_ctx = self.build_prompt(chain, kg_context)
        if self.llm_fn is not None:
            parsed = self.parse_llm_output(self.llm_fn(prompt))
            question, answer = parsed["question"], parsed["answer"]
        else:
            question, answer = _template_generate(prompt_ctx)

        required_tools = [s["tool"] for s in chain["steps"]]
        toolkg_edges_used = self._toolkg_edges_used(required_tools)
        base = {
            "question": question,
            "answer_draft": answer,
            "required_tools": required_tools,
            "chain": chain["steps"],
            "entity_refs": prompt_ctx["entity_refs"],
            "provenance": {
                "source": "toolgrad_trace",
                "chain_id": chain["chain_id"],
                "trace_seed": trace_seed,
                "toolkg_edges_used": toolkg_edges_used,
                "kg_entity_count": len(_entity_names(kg_context)),
            },
        }
        return [base] * 1 if self.max_pairs_per_chain == 1 else [dict(base) for _ in range(self.max_pairs_per_chain)]

    def process(
        self, samples: Sequence[dict], tracers: Optional[Sequence[Optional[dict]]] = None
    ) -> Tuple[List[dict], dict]:
        """Convert workflow samples into QA pairs.

        Mirrors GraphGen's ``process(batch) -> (results, stats)`` contract.
        ``tracers[i]`` (optional) supplies provenance for ``samples[i]``.
        """
        results: List[dict] = []
        chains_seen = 0
        chains_empty = 0
        tracers = tracers or [None] * len(samples)
        for sample, tracer in zip(samples, tracers):
            seed = tracer.get("seed") if isinstance(tracer, dict) else None
            chains = extract_chains(sample)
            if not chains:
                chains_empty += 1
            for chain in chains:
                chains_seen += 1
                results.extend(self.generate_for_chain(chain, self.kg_context, seed))
        stats = {
            "num_samples": len(samples),
            "num_chains": chains_seen,
            "num_chains_empty": chains_empty,
            "num_qa_pairs": len(results),
        }
        return results, stats

    # -- helpers ----------------------------------------------------------

    def _toolkg_edges_used(self, tools: List[str]) -> int:
        """Count consecutive tool pairs present as edges in the ToolKG."""
        if self.toolkg is None or len(tools) < 2:
            return 0
        return sum(
            1 for a, b in zip(tools, tools[1:]) if self.toolkg.has_edge(a, b)
        )

    @staticmethod
    def to_chatml(qa: dict) -> List[dict]:
        """Render a QA pair in GraphGen's SFT format (ChatML).

        The user turn carries the question; the assistant turn replays the
        executed chain and gives the drafted answer — the standard
        tool-use SFT shape Phase 4 consumes.
        """
        chain_lines = [
            f"{i}. {s['tool']}({json.dumps(s['tool_input'], ensure_ascii=False)}) -> {s['result_preview']}"
            for i, s in enumerate(qa["chain"], 1)
        ]
        tools = ", ".join(qa["required_tools"])
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions by "
                    "calling tools. Use the tools listed; do not rely on "
                    "general knowledge."
                ),
            },
            {
                "role": "user",
                "content": f"{qa['question']}\n\nAvailable tools: {tools}",
            },
            {
                "role": "assistant",
                "content": (
                    "Executed chain:\n" + "\n".join(chain_lines)
                    + f"\n\nAnswer: {qa['answer_draft']}"
                ),
            },
        ]
