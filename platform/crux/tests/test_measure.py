from crux.measure import first_entity_id, measure_cells
import json


class RecordingClient:
    def __init__(self, entities_text: str):
        self.entities_text = entities_text
        self.calls = []

    def start_session(self, kb_ids):
        return {"session_id": "s1", "kb_ids": list(kb_ids)}

    def ask(self, *, kb_id, tool, arguments, session_id, session_kb_ids=None):
        self.calls.append((tool, arguments))
        if tool == "search_chunks":
            saved = 4
            text = "document hit"
        elif tool == "find_entities":
            saved = 2
            text = self.entities_text
        elif tool == "entity_facts":
            saved = 9
            text = "publishes lineage"
        else:
            raise AssertionError(tool)
        return {
            "text": text,
            "compressed": {
                "tokens_before": 20,
                "tokens_after": 20 - saved,
                "tokens_saved": saved,
                "skipped": False,
            },
        }


def test_first_entity_id_reads_find_entities_row():
    text = (
        "01a07cbf-668c-73b1-a559-1e5002f5710f | Azure Purview | Service | 4 facts"
    )
    assert first_entity_id(text) == "01a07cbf-668c-73b1-a559-1e5002f5710f"
    assert first_entity_id("No matching entities.") is None


def test_first_entity_id_uses_first_column_only():
    text = (
        "01a07cbf-668c-73b1-a559-1e5002f5710f | Label | see "
        "f47ac10b-58cc-4372-a567-0e02b2c3d479 | 4 facts"
    )
    assert first_entity_id(text) == "01a07cbf-668c-73b1-a559-1e5002f5710f"


def test_measure_records_all_four_cells_without_caller_bypass():
    client = RecordingClient(
        "01a07cbf-668c-73b1-a559-1e5002f5710f | Azure Purview | Service | 4 facts"
    )
    report = measure_cells(client, kb_id="kb-production", query="write path")
    assert report["hypothesis"] is True
    assert report["cells"]["B_document_compress_on"]["tokens_saved"] == 4
    assert report["cells"]["D_graph_compress_on"]["tokens_saved"] == 9
    assert report["cells"]["A_document_compress_off"]["tokens"] == 20
    assert report["cells"]["A_document_compress_off"]["derived_from"] == "tokens_before"
    assert report["cells"]["A_document_compress_off"]["no_caller_bypass"] is True
    assert report["cells"]["C_graph_compress_off"]["tokens"] == 20
    assert report["cells"]["C_graph_compress_off"]["no_caller_bypass"] is True
    assert report["graph"]["entity_id"] == "01a07cbf-668c-73b1-a559-1e5002f5710f"
    assert "A_document_compress_off" not in report["skipped"]
    assert "C_graph_compress_off" not in report["skipped"]
    tools = [call[0] for call in client.calls]
    assert tools == ["search_chunks", "find_entities", "entity_facts"]
    assert client.calls[2][1] == {"entity_id": "01a07cbf-668c-73b1-a559-1e5002f5710f"}
    assert "query" not in client.calls[2][1]
    assert "bypass" not in json.dumps(client.calls)


def test_measure_skips_graph_cell_when_no_entities():
    client = RecordingClient("No matching entities.")
    report = measure_cells(client, kb_id="kb-staging", query="engineering")
    assert report["cells"]["B_document_compress_on"]["tokens_saved"] == 4
    assert report["cells"]["D_graph_compress_on"] is None
    assert report["cells"]["C_graph_compress_off"] is None
    assert "no_graph_coverage" in report["skipped"]["D_graph_compress_on"]
    assert "no_graph_coverage" in report["skipped"]["C_graph_compress_off"]
    tools = [call[0] for call in client.calls]
    assert tools == ["search_chunks", "find_entities"]
