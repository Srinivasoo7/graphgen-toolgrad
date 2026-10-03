"""Exporters: pure transforms from ScenarioPackage to downstream formats.

Each exporter is one function: package(s) in, the consumer's shape out.
No LLM, no network, no side effects beyond the optional file write.
"""

from bridge.exporters.copilot_eval import (
    eval_row,
    to_jsonl,
    write_jsonl,
)

__all__ = ["eval_row", "to_jsonl", "write_jsonl"]
