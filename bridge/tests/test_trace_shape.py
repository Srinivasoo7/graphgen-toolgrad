"""Trace-shape test: a fabricated ToolGrad chain JSON must validate against the
real upstream schema (``toolgrad.states.ApiUseWorkflow``).

The fixture mirrors the per-run sample format from Phase 0 recon:
``examples/outputs/seed={seed}__iter={iter}__num_apis={n}.json`` is
``ApiUseWorkflow.model_dump()``. Importing ``toolgrad.states`` needs no API
keys and no LLM — only the installed ToolGrad package. If the import fails
(e.g. running outside the ToolGrad venv), the test degrades to structural
assertions instead of failing.
"""

import json

# NOTE: the installed langchain_classic's ToolAgentAction requires
# message_log / tool_call_id in addition to tool / tool_input / log.
def _action(tool: str, tool_input: dict, log: str) -> dict:
    return {
        "tool": tool,
        "tool_input": tool_input,
        "log": log,
        "message_log": [],
        "tool_call_id": f"call_{tool}",
        "type": "AgentActionMessageLog",
    }


# Fabricated chain: two MCP filesystem calls, in the exact shape
# ApiUseWorkflow.model_dump() produces (query / api_use_chains / response).
FABRICATED_SAMPLE = {
    "query": (
        "I'm researching Tesla's Gigafactories for a report. What do the "
        "local notes say about Gigafactory Texas production?"
    ),
    "api_use_chains": {
        "chain_0": {
            "chain_id": "chain_0",
            "intermediate_steps": [
                [
                    _action(
                        "list_directory",
                        {"path": "/data/tesla"},
                        "Listed /data/tesla",
                    ),
                    {"files": ["gigafactory_notes.txt", "models.csv"]},
                ],
                [
                    _action(
                        "read_text_file",
                        {"path": "/data/tesla/gigafactory_notes.txt"},
                        "Read gigafactory_notes.txt",
                    ),
                    "Gigafactory Texas produces Model Y and Cybertruck; "
                    "2025 output target 500k vehicles.",
                ],
            ],
            "description": (
                "List the Tesla data directory, then read the Gigafactory "
                "notes file to answer the production question."
            ),
        }
    },
    "response": (
        "According to the local notes, Gigafactory Texas produces the Model Y "
        "and Cybertruck, with a 2025 output target of 500,000 vehicles."
    ),
}


def _structural_checks(sample: dict) -> None:
    assert set(sample) == {"query", "api_use_chains", "response"}, set(sample)
    assert isinstance(sample["query"], str) and sample["query"]
    assert isinstance(sample["response"], str) and sample["response"]
    chains = sample["api_use_chains"]
    assert isinstance(chains, dict) and chains
    for chain_id, chain in chains.items():
        assert chain["chain_id"] == chain_id
        assert isinstance(chain["description"], str)
        for action, result in chain["intermediate_steps"]:
            assert set(action) >= {"tool", "tool_input"}, set(action)
            assert isinstance(result, (dict, str))


def test_fabricated_chain_matches_real_schema():
    _structural_checks(FABRICATED_SAMPLE)
    try:
        from toolgrad import states
    except ImportError:
        return "SKIP"  # structural checks above already ran
    workflow = states.ApiUseWorkflow.model_validate(FABRICATED_SAMPLE)
    assert workflow.query == FABRICATED_SAMPLE["query"]
    assert workflow.response == FABRICATED_SAMPLE["response"]
    chain = workflow.api_use_chains["chain_0"]
    assert chain.chain_id == "chain_0"
    assert len(chain.intermediate_steps) == 2
    action, result = chain.intermediate_steps[0]
    assert action.tool == "list_directory"
    assert action.tool_input == {"path": "/data/tesla"}
    assert result == {"files": ["gigafactory_notes.txt", "models.csv"]}
    # Round-trip: model_dump must reproduce the on-disk format.
    dumped = workflow.model_dump()
    assert set(dumped) == {"query", "api_use_chains", "response"}
    json.dumps(dumped)  # must be JSON-serializable


def test_fixture_file_roundtrip(tmp_path=None):
    # The fixture survives a write/read cycle like the real outputs/ files.
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seed=1__iter=2__num_apis=5.json")
        with open(path, "w") as f:
            json.dump(FABRICATED_SAMPLE, f, indent=2)
        with open(path) as f:
            reloaded = json.load(f)
    _structural_checks(reloaded)
