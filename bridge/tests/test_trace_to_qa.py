"""Tests for bridge/trace_to_qa.py (keyless)."""

import networkx as nx

from bridge import kg_context_exporter, toolkg_builder, trace_to_qa
from bridge.tests.test_toolkg_builder import _catalog, _HINTS
from bridge.tests.test_trace_shape import FABRICATED_SAMPLE


def _kg_context():
    g = nx.Graph()
    g.add_node("n1", entity_name="Tesla", entity_type="ORG",
               description="Electric vehicle manufacturer")
    g.add_node("n2", entity_name="Gigafactory Texas", entity_type="FACILITY",
               description="Tesla factory in Austin producing Model Y and Cybertruck")
    g.add_node("n3", entity_name="Model Y", entity_type="PRODUCT",
               description="Tesla crossover SUV")
    g.add_edge("n1", "n2", description="Tesla operates Gigafactory Texas")
    g.add_edge("n2", "n3", description="Gigafactory Texas produces Model Y")
    return kg_context_exporter.export_kg_context(g)


def _toolkg():
    return toolkg_builder.build_toolkg(_catalog(), output_hints=_HINTS)


def test_extract_chains_normalizes_steps():
    chains = trace_to_qa.extract_chains(FABRICATED_SAMPLE)
    assert len(chains) == 1
    chain = chains[0]
    assert chain["chain_id"] == "chain_0"
    assert [s["tool"] for s in chain["steps"]] == ["list_directory", "read_text_file"]
    assert chain["steps"][0]["tool_input"] == {"path": "/data/tesla"}
    assert "Gigafactory Texas" in chain["steps"][1]["result_preview"]


def test_ground_entities_matches_kg_names():
    chains = trace_to_qa.extract_chains(FABRICATED_SAMPLE)
    refs = trace_to_qa.ground_entities(chains[0], _kg_context())
    assert "Gigafactory Texas" in refs
    assert "Tesla" in refs  # appears in the /data/tesla path
    # every ref is a real KG entity name
    names = {e["name"] for e in _kg_context()["entities"]}
    assert set(refs) <= names


def test_operator_process_emits_grounded_qa():
    op = trace_to_qa.TraceToQAOperator(kg_context=_kg_context(), toolkg=_toolkg())
    results, stats = op.process([FABRICATED_SAMPLE], [{"seed": 123}])
    assert stats["num_chains"] == 1
    assert len(results) == 1
    qa = results[0]
    assert set(qa) == {"question", "answer_draft", "required_tools",
                       "chain", "entity_refs", "provenance"}
    # question names real KG entities
    assert any(e in qa["question"] for e in qa["entity_refs"])
    assert qa["entity_refs"], "question must be anchored to at least one KG entity"
    # tools come from the ToolKG
    assert set(qa["required_tools"]) <= set(_toolkg().nodes)
    assert qa["required_tools"] == ["list_directory", "read_text_file"]
    # provenance is complete
    prov = qa["provenance"]
    assert prov["source"] == "toolgrad_trace"
    assert prov["chain_id"] == "chain_0"
    assert prov["trace_seed"] == 123
    # list_directory -> read_text_file is a real composability edge
    assert prov["toolkg_edges_used"] == 1


def test_llm_fn_is_called_with_kg_and_chain():
    seen = {}

    def stub(prompt: str) -> str:
        seen["prompt"] = prompt
        return "QUESTION: Which models does Gigafactory Texas build?\nANSWER: Model Y and Cybertruck."

    op = trace_to_qa.TraceToQAOperator(llm_fn=stub, kg_context=_kg_context())
    results, _ = op.process([FABRICATED_SAMPLE])
    qa = results[0]
    assert qa["question"] == "Which models does Gigafactory Texas build?"
    assert qa["answer_draft"] == "Model Y and Cybertruck."
    # the prompt actually carried the KG context and the chain
    assert "Gigafactory Texas" in seen["prompt"]
    assert "read_text_file" in seen["prompt"]


def test_stubbed_llm_is_deterministic():
    def stub(prompt: str) -> str:
        return "QUESTION: Q?\nANSWER: A."

    op = trace_to_qa.TraceToQAOperator(llm_fn=stub, kg_context=_kg_context())
    r1, _ = op.process([FABRICATED_SAMPLE])
    r2, _ = op.process([FABRICATED_SAMPLE])
    assert r1[0]["question"] == r2[0]["question"]
    assert r1[0]["answer_draft"] == r2[0]["answer_draft"]


def test_parse_llm_output_degrades_gracefully():
    parsed = trace_to_qa.TraceToQAOperator.parse_llm_output("Just a question, no format")
    assert parsed["question"] == "Just a question, no format"
    assert parsed["answer"]  # fallback answer, never empty
    parsed = trace_to_qa.TraceToQAOperator.parse_llm_output("")
    assert parsed["question"] == ""


def test_process_skips_empty_chains():
    sample = dict(FABRICATED_SAMPLE)
    sample["api_use_chains"] = {"chain_0": {"chain_id": "chain_0",
                                            "intermediate_steps": [],
                                            "description": "empty"}}
    op = trace_to_qa.TraceToQAOperator(kg_context=_kg_context())
    results, stats = op.process([sample])
    assert results == []
    assert stats["num_chains_empty"] == 1


def test_to_chatml_shape():
    op = trace_to_qa.TraceToQAOperator(kg_context=_kg_context(), toolkg=_toolkg())
    (qa,), _ = op.process([FABRICATED_SAMPLE])
    turns = trace_to_qa.TraceToQAOperator.to_chatml(qa)
    roles = [t["role"] for t in turns]
    assert roles == ["system", "user", "assistant"]
    assert qa["question"] in turns[1]["content"]
    assert "list_directory" in turns[1]["content"]  # available tools listed
    assert "Executed chain:" in turns[2]["content"]
    assert qa["answer_draft"] in turns[2]["content"]


def test_ground_entities_includes_step_tool_names():
    # A domain KG may model the tools themselves as entities; a chain that
    # invokes such a tool is anchored to it even when inputs/results are bare.
    g = nx.Graph()
    g.add_node("n1", entity_name="list_directory", entity_type="TOOL")
    kg = kg_context_exporter.export_kg_context(g)
    chain = {
        "chain_id": "c0",
        "steps": [
            {"tool": "list_directory", "tool_input": {"path": "."},
             "result_preview": "3 files"},
        ],
    }
    assert trace_to_qa.ground_entities(chain, kg) == ["list_directory"]


def test_prompt_frames_request_fulfilled_by_chain():
    # Regression test: the 2026-09-20 ITSM run produced "What is the status
    # of ticket INC-1042?" for a chain whose only step *creates* the ticket.
    # The prompt must demand a request the chain fulfills, never a lookup
    # about a record the chain creates.
    op = trace_to_qa.TraceToQAOperator(kg_context=_kg_context())
    chains = trace_to_qa.extract_chains(FABRICATED_SAMPLE)
    prompt, _ = op.build_prompt(chains[0], _kg_context())
    assert "fulfilled by executing the tool chain" in prompt
    assert "Never ask about" in prompt
    assert "already existed" in prompt


def test_template_fallback_is_request_style():
    # The keyless fallback must also read as a request, not a lookup.
    op = trace_to_qa.TraceToQAOperator(kg_context=_kg_context())  # llm_fn=None
    (qa,), _ = op.process([FABRICATED_SAMPLE])
    assert qa["question"].startswith(
        "Use the list_directory, read_text_file tools to list")
    assert any(e in qa["question"] for e in qa["entity_refs"])


def test_request_verb_mapping():
    assert trace_to_qa._request_verb("create_ticket") == "create"
    assert trace_to_qa._request_verb("get_kb_article") == "look up"
    assert trace_to_qa._request_verb("update_ticket") == "update"
    assert trace_to_qa._request_verb("frobnicate_widgets") == "use"
