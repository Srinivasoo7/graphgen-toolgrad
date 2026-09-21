"""Tests for bridge/refinement_loop.py (keyless)."""

import json
import os
import tempfile

from bridge import refinement_loop, trace_to_qa
from bridge.refinement_loop import RefineConfig, RefinementLoop
from bridge.tests import _fixtures


def _loop(**kwargs):
    kwargs.setdefault("kg_context", _fixtures.kg_context())
    kwargs.setdefault("toolkg", _fixtures.toolkg())
    kwargs.setdefault("executor", _fixtures.executor())
    return RefinementLoop(**kwargs)


def test_loop_all_pass_first_iteration():
    loop = _loop()  # llm_fn=None -> template generation
    kept, ledger = loop.run([_fixtures.sample()], [{"seed": 123}])
    assert ledger["stopped_reason"] == "all_pass"
    assert len(ledger["iterations"]) == 1
    assert ledger["iterations"][0]["num_pairs"] == 1
    assert ledger["iterations"][0]["num_kept"] == 1
    assert len(kept) == 1
    # scores are attached for the SFT mix
    assert kept[0]["verification"]["chain_valid"] is True
    assert kept[0]["coverage"] == 1.0


def test_loop_no_llm_stops_after_filtering():
    ex = _fixtures.executor()
    # drop read_text_file: the chain can no longer verify
    ex2 = type(ex)()
    ex2.register(
        "list_directory",
        {"type": "object", "properties": {"path": {"type": "string"}},
         "required": ["path"]},
        lambda ti: {"files": []},
    )
    loop = _loop(executor=ex2)  # llm_fn=None: no honest refinement possible
    kept, ledger = loop.run([_fixtures.sample()])
    assert ledger["stopped_reason"] == "no_llm_refinement"
    assert kept == []
    assert ledger["iterations"][0]["num_rejected"] == 1


def test_refinement_improves_with_stubbed_llm():
    calls = []

    def stub(prompt: str) -> str:
        calls.append(prompt)
        if "Critique of your previous attempt" in prompt:
            return ("QUESTION: What does Gigafactory Texas produce?\n"
                    "ANSWER: Model Y and Cybertruck, per the notes file.")
        return "QUESTION: What do the local records say?\nANSWER: Some records."

    loop = _loop(llm_fn=stub, config=RefineConfig(max_iterations=3))
    kept, ledger = loop.run([_fixtures.sample()], [{"seed": 7}])
    assert ledger["stopped_reason"] == "all_pass"
    assert len(ledger["iterations"]) == 2
    assert ledger["iterations"][0]["num_rejected"] == 1  # unanchored question
    assert ledger["iterations"][0]["num_refined"] == 1
    assert len(kept) == 1
    assert "Gigafactory Texas" in kept[0]["question"]
    assert kept[0]["provenance"]["refinement_attempt"] == 1
    assert kept[0]["provenance"]["critique"]  # critique was recorded


def test_critique_lists_concrete_issues():
    ex = _fixtures.executor()
    qa = {
        "question": "Q?",
        "answer_draft": "A.",
        "required_tools": ["list_directory", "nope_tool"],
        "chain": [
            {"tool": "list_directory", "tool_input": {"path": "/x"},
             "result_preview": "..."},
            {"tool": "nope_tool", "tool_input": {}, "result_preview": "..."},
        ],
        "entity_refs": [],
        "provenance": {},
    }
    score = refinement_loop.score_qa(
        qa, ex, _fixtures.toolkg(), _fixtures.kg_context())
    issues = refinement_loop.critique_qa(qa, score, RefineConfig())
    assert any("nope_tool" in i for i in issues)
    assert any("entity" in i.lower() for i in issues)


def test_score_qa_entity_grounding_requires_question_mention():
    ex = _fixtures.executor()
    op = trace_to_qa.TraceToQAOperator(
        kg_context=_fixtures.kg_context(), toolkg=_fixtures.toolkg())
    (qa,), _ = op.process([_fixtures.sample()])
    # template question names the entities -> grounded
    score = refinement_loop.score_qa(qa, ex, _fixtures.toolkg(), _fixtures.kg_context())
    assert score["entity_grounded"] is True
    qa2 = dict(qa, question="What do the local records say?")
    score2 = refinement_loop.score_qa(qa2, ex, _fixtures.toolkg(), _fixtures.kg_context())
    assert score2["entity_grounded"] is False  # refs exist but question is generic


def test_filter_pairs_splits_on_thresholds():
    config = RefineConfig(min_coverage=0.5)
    good = ({"question": "g"}, {"chain_valid": True, "coverage": 1.0,
                                "entity_grounded": True})
    bad = ({"question": "b"}, {"chain_valid": True, "coverage": 0.1,
                               "entity_grounded": True})
    kept, rejected = refinement_loop.filter_pairs([good, bad], config)
    assert [q for q, _ in kept] == [{"question": "g"}]
    assert rejected == [{"question": "b"}]


def test_ledger_roundtrip():
    loop = _loop()
    kept, ledger = loop.run([_fixtures.sample()])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        refinement_loop.save_ledger(ledger, path)
        back = refinement_loop.load_ledger(path)
    assert back["stopped_reason"] == ledger["stopped_reason"]
    assert back["iterations"][0]["num_kept"] == 1
    assert json.dumps(back)  # JSON-serializable


def test_loop_requires_executor():
    loop = RefinementLoop(kg_context=_fixtures.kg_context())
    try:
        loop.run([_fixtures.sample()])
    except ValueError:
        return
    raise AssertionError("expected ValueError without an executor")


def test_filter_fn_rejects_and_annotates():
    seen = []

    def reject_all(qa, score):
        seen.append(qa)
        qa["jev"] = {"verdict": "reject"}
        return False

    loop = _loop(filter_fn=reject_all)
    kept, ledger = loop.run([_fixtures.sample()], [{"seed": 123}])
    assert kept == []
    metrics = ledger["iterations"][0]
    assert metrics["num_semantic_filtered"] == 1
    assert metrics["num_rejected"] == 1
    assert ledger["stopped_reason"] != "all_pass"
    assert seen[0]["jev"]["verdict"] == "reject"


def test_filter_fn_accept_keeps_all_pass():
    loop = _loop(filter_fn=lambda qa, score: True)
    kept, ledger = loop.run([_fixtures.sample()], [{"seed": 123}])
    assert len(kept) == 1
    assert ledger["stopped_reason"] == "all_pass"
    assert ledger["iterations"][0]["num_semantic_filtered"] == 0


def test_filter_fn_exception_fails_open():
    def boom(qa, score):
        raise RuntimeError("judge exploded")

    loop = _loop(filter_fn=boom)
    kept, ledger = loop.run([_fixtures.sample()], [{"seed": 123}])
    assert len(kept) == 1  # a broken filter must not kill the run
    assert ledger["stopped_reason"] == "all_pass"


def test_critique_names_candidate_entities():
    # The entity critique must name the actual KG candidates so the LLM can
    # comply; a vague "name a real entity" is not actionable.
    score = {
        "chain_valid": True,
        "verification": {"chain_valid": True},
        "coverage": 1.0,
        "entity_grounded": False,
        "entity_refs": ["list_directory_with_sizes"],
        "required_tools": ["list_directory_with_sizes"],
        "question_tokens": 10,
        "answer_tokens": 10,
    }
    qa = {"question": "What do the records say?", "answer_draft": "A."}
    issues = refinement_loop.critique_qa(qa, score, RefineConfig())
    assert any("list_directory_with_sizes" in i for i in issues)


def test_critique_entity_fallback_without_refs():
    score = {
        "chain_valid": True,
        "verification": {"chain_valid": True},
        "coverage": 1.0,
        "entity_grounded": False,
        "entity_refs": [],
        "required_tools": [],
        "question_tokens": 10,
        "answer_tokens": 10,
    }
    qa = {"question": "Q?", "answer_draft": "A."}
    issues = refinement_loop.critique_qa(qa, score, RefineConfig())
    assert any("entity" in i.lower() for i in issues)
