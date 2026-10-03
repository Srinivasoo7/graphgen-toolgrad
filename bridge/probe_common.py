"""Shared plumbing for the tabular probe stage (backend-agnostic).

Table loading, column statistics, scoring, splitting, and the
findings -> corpus narration live here so probe backends (Kumo Tabular
today) stay thin. Findings flow one way: measured aggregates become
documents; raw rows never leave the probe.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class ProbeError(RuntimeError):
    """The probe stage cannot run (bad tables, missing package, bad target)."""


# ---------------------------------------------------------------------------
# loading: CSV / JSON tables -> rows of coerced scalars
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """Coerce a raw cell to int / float / str / None (CSV arrives as str)."""
    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            continue
    return text


def load_tables_dir(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load every ``.csv`` / ``.json`` table in a directory.

    JSON tables are a list of objects. Returns ``{table_name: rows}`` with
    cell values coerced to numbers where they parse as numbers.
    """
    tables: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(path):
        raise ProbeError(f"tables_dir not found: {path}")
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        name, ext = os.path.splitext(fname)
        if ext.lower() == ".csv":
            with open(fpath, newline="", encoding="utf-8") as fh:
                rows = [
                    {k: _coerce(v) for k, v in row.items() if k}
                    for row in csv.DictReader(fh)
                ]
        elif ext.lower() == ".json":
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list) or not all(
                isinstance(r, dict) for r in data
            ):
                raise ProbeError(f"{fname}: JSON table must be a list of objects")
            rows = [{k: _coerce(v) for k, v in r.items()} for r in data]
        else:
            continue
        tables[name] = rows
    return tables


# ---------------------------------------------------------------------------
# column statistics (pure python — no model involved)
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def column_stats(rows: Sequence[Dict[str, Any]], column: str) -> Dict[str, Any]:
    """Type, cardinality, missingness, and headline values for one column."""
    present = [r.get(column) for r in rows if r.get(column) is not None]
    counts: Dict[Any, int] = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    numeric = bool(present) and all(_is_number(v) for v in present)
    stats: Dict[str, Any] = {
        "name": column,
        "type": "numeric" if numeric else "categorical",
        "cardinality": len(counts),
        "missing_frac": round(1.0 - len(present) / len(rows), 4) if rows else 0.0,
    }
    if numeric and present:
        stats["min"] = min(present)
        stats["max"] = max(present)
        stats["mean"] = round(sum(present) / len(present), 4)
    elif present:
        top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:5]
        stats["top_values"] = [{"value": str(v), "count": c} for v, c in top]
    return stats


# ---------------------------------------------------------------------------
# scoring + splitting (estimator-agnostic)
# ---------------------------------------------------------------------------


def _score(task: str, y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if task == "classification":
        hits = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return round(hits / len(y_true), 4) if y_true else 0.0
    mean = sum(y_true) / len(y_true) if y_true else 0.0
    ss_tot = sum((v - mean) ** 2 for v in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0


def _split(n: int, seed: int) -> Tuple[List[int], List[int]]:
    """Deterministic 80/20 split; tiny tables score on their own train set."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    if n < 10:
        return idx, idx
    cut = max(1, int(0.8 * n))
    return idx[:cut], idx[cut:]


# ---------------------------------------------------------------------------
# narration: findings -> corpus documents (findings only, never raw rows)
# ---------------------------------------------------------------------------


def findings_to_markdown(table_findings: Dict[str, Any]) -> str:
    """Render one table's findings as a deterministic markdown document."""
    lines = [
        f"# Table: {table_findings['table']}",
        "",
        f"{table_findings['num_rows']} rows"
        + (
            f" (probed on a sample of {table_findings['sampled_rows']})"
            if table_findings["sampled_rows"] != table_findings["num_rows"]
            else ""
        )
        + ".",
        "",
        "## Columns",
        "",
    ]
    for col in table_findings["columns"]:
        detail = f"{col['type']}, {col['cardinality']} distinct"
        if col["missing_frac"]:
            detail += f", {col['missing_frac']:.1%} missing"
        if col["type"] == "numeric" and "min" in col:
            detail += f", range {col['min']} to {col['max']}, mean {col['mean']}"
        elif col.get("top_values"):
            tops = ", ".join(
                f"{t['value']} ({t['count']})" for t in col["top_values"]
            )
            detail += f"; most common: {tops}"
        lines.append(f"- {col['name']}: {detail}")
    if table_findings["targets"]:
        lines += ["", "## Measured relationships", ""]
        for tgt in table_findings["targets"]:
            metric = "accuracy" if tgt["task"] == "classification" else "R2"
            lines.append(
                f"- {tgt['column']} ({tgt['task']}): holdout {metric} "
                f"{tgt['holdout_score']} vs baseline {tgt['baseline_score']}."
            )
            drivers = [i for i in tgt["importances"] if i["importance"] > 0]
            if drivers:
                named = ", ".join(
                    f"{d['feature']} ({d['importance']})" for d in drivers[:5]
                )
                lines.append(f"  Strongest drivers: {named}.")
    return "\n".join(lines) + "\n"


_NARRATE_PROMPT = """\
Rewrite the measured findings below about an enterprise data table as a
short domain brief for a knowledge graph. Use ONLY facts present in the
findings; do not invent values, entities, or relationships. Name the
business entities and the rules the findings support.

--- findings ---
{doc}
"""


def write_tables_corpus(
    findings: Dict[str, Any],
    out_dir: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Write findings documents (+ optional LLM narrative briefs) to a dir.

    The deterministic findings doc is always written — it is the auditable
    source. When ``llm_fn`` is given, a prose brief narrated from the
    findings (never from raw rows) is written alongside it. Returns the
    list of written paths. Also writes ``findings.json`` for audit.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for table_findings in findings["tables"]:
        doc = findings_to_markdown(table_findings)
        path = os.path.join(out_dir, f"{table_findings['table']}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        paths.append(path)
        if llm_fn is not None:
            brief = llm_fn(_NARRATE_PROMPT.format(doc=doc))
            npath = os.path.join(out_dir, f"{table_findings['table']}_narrative.md")
            with open(npath, "w", encoding="utf-8") as fh:
                fh.write(brief if brief.endswith("\n") else brief + "\n")
            paths.append(npath)
    with open(os.path.join(out_dir, "findings.json"), "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2)
    return paths
