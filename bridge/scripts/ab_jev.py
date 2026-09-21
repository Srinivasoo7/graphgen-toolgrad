"""A/B: heuristic quality gates vs the Jev-backed filter (live experiment).

Builds a small labeled set of QA pairs spanning the heuristic gates'
axes (chain validity, ToolKG coverage, entity grounding) plus cases the
heuristics cannot see (empty / contradictory / fluff answers), scores them
with the keyless pipeline, then runs both filters and compares.

Phase 2 re-decides the interesting pairs to measure verdict stability,
since Jev is non-deterministic across calls.

Usage (from the repo root)::

    python3 bridge/scripts/ab_jev.py [--out report.json] [--stability-repeats 2]

Writes a JSON report (default: bridge/scripts/ab_jev_report.json) and prints
a summary table. Costs a few hundred input tokens of Jev calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bridge.jev_filter import JevDecider, jev_filter_pairs
from bridge.refinement_loop import RefineConfig, filter_pairs, score_qa
from bridge.tests import _fixtures

PRICE_PER_MTOK = 0.042  # USD, TypeSafe launch pricing (input; output free)


def _step(tool, tool_input, preview="ok"):
    return {"tool": tool, "tool_input": tool_input, "result_preview": preview}


GOOD_CHAIN = [
    _step("list_directory", {"path": "/notes"}),
    _step("read_text_file", {"path": "/notes/gigafactory_notes.txt"},
          "Gigafactory Texas produces Model Y and Cybertruck."),
]


def _qa(question, answer, chain, refs):
    return {
        "question": question,
        "answer_draft": answer,
        "entity_refs": refs,
        "required_tools": [s["tool"] for s in chain],
        "chain": chain,
    }


def build_pairs():
    raw = [
        ("good", _qa("What does Gigafactory Texas produce?",
                     "Model Y and Cybertruck, per the notes file.",
                     GOOD_CHAIN, ["Gigafactory Texas"])),
        ("good_model_y", _qa("Which Tesla model is built at Gigafactory Texas?",
                             "Model Y, according to the factory notes.",
                             GOOD_CHAIN, ["Model Y", "Gigafactory Texas"])),
        ("good_tesla", _qa("What does Tesla manufacture at Gigafactory Texas?",
                           "Tesla builds the Model Y and Cybertruck there.",
                           GOOD_CHAIN, ["Tesla", "Gigafactory Texas"])),
        ("ungrounded", _qa("What do the local records say?",
                           "The records mention production figures.",
                           GOOD_CHAIN, [])),
        ("wrong_entity", _qa("What does Atlantis produce?",
                             "Unknown.",
                             GOOD_CHAIN, ["Atlantis"])),
        ("partial_ground", _qa("What does the factory produce?",
                               "Model Y and Cybertruck.",
                               GOOD_CHAIN, ["Gigafactory Texas"])),
        ("missing_tool", _qa("What else does Gigafactory Texas produce?",
                             "Model Y.",
                             GOOD_CHAIN + [_step("delete_database", {})],
                             ["Gigafactory Texas"])),
        ("bad_args", _qa("What is made at Gigafactory Texas?",
                         "Model Y.",
                         [_step("list_directory", {"path": "/notes"}),
                          _step("read_text_file", {})],  # missing required path
                         ["Gigafactory Texas"])),
        ("single_step", _qa("Which vehicles come from Gigafactory Texas?",
                            "Model Y and Cybertruck.",
                            [_step("read_text_file",
                                   {"path": "/notes/gigafactory_notes.txt"})],
                            ["Gigafactory Texas"])),
        ("empty_answer", _qa("What is Gigafactory Texas known for?",
                             "   ",
                             GOOD_CHAIN, ["Gigafactory Texas"])),
        ("contradictory_answer", _qa(
            "What is produced at Gigafactory Texas?",
            "The Mars colony produces oxygen and water ice.",
            GOOD_CHAIN, ["Gigafactory Texas"])),
        ("fluff_answer", _qa("What comes out of Gigafactory Texas?",
                             "Some records exist about production.",
                             GOOD_CHAIN, ["Gigafactory Texas"])),
    ]
    for name, qa in raw:
        qa["pair_id"] = name  # unique key; the filter copies qa dicts shallowly
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ab_jev_report.json"))
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--stability-repeats", type=int, default=2,
                    help="extra decide() calls per heuristic-accepted pair")
    args = ap.parse_args()

    kg_context = _fixtures.kg_context()
    toolkg = _fixtures.toolkg()
    executor = _fixtures.executor()
    config = RefineConfig()

    pairs = build_pairs()
    scored = [(qa, score_qa(qa, executor, toolkg, kg_context)) for _, qa in pairs]

    h_kept, h_rejected = filter_pairs(scored, config)
    h_verdict = {qa["pair_id"]: "accept" for qa, _ in h_kept}
    h_verdict.update({qa["pair_id"]: "reject" for qa in h_rejected})

    # Phase 1: agreement — one live Jev decision per pair through the real filter
    decider = JevDecider(model=args.model)
    j_kept, j_rejected = jev_filter_pairs(scored, config, decider)
    annotations = {}
    for qa, _ in j_kept:
        annotations[qa["pair_id"]] = qa["jev"]
    for qa in j_rejected:
        annotations[qa["pair_id"]] = qa["jev"]

    rows = []
    for (qa, score), (name, _) in zip(scored, pairs):
        ann = annotations[name]
        rows.append({
            "name": name,
            "heuristic": h_verdict[name],
            "jev": ann["verdict"],
            "agree": h_verdict[name] == ann["verdict"],
            "p_accept": ann["confidence"],
            "grounded_noul": ann["grounded"],
            "latency_ms": ann["latency_ms"],
            "source": ann["source"],
            "error": ann["error"],
            "input_tokens": ann["input_tokens"],
            "keyless": {
                "chain_valid": score["chain_valid"],
                "coverage": round(score["coverage"], 3),
                "entity_grounded": score["entity_grounded"],
            },
        })

    # Phase 2: stability — re-decide heuristic-accepted pairs, watch P(accept)
    stability = []
    for (qa, score), (name, _) in zip(scored, pairs):
        if h_verdict[name] != "accept":
            continue
        ps = [annotations[name]["confidence"]]
        for _ in range(args.stability_repeats):
            d = decider.decide(dict(qa), score, config)
            ps.append(d["confidence"])
        flips = sum(1 for p in ps[1:] if (p >= 0.5) != (ps[0] >= 0.5))
        stability.append({
            "name": name,
            "p_accept_runs": [round(p, 3) for p in ps],
            "spread": round(max(ps) - min(ps), 3),
            "flips": flips,
        })

    agree = sum(1 for r in rows if r["agree"])
    lat = sorted(r["latency_ms"] for r in rows)
    in_tokens = sum(r["input_tokens"] or 0 for r in rows)
    # stability repeats also consumed tokens; estimate from mean per-call
    mean_tokens = in_tokens / len(rows) if rows else 0
    total_tokens = in_tokens + int(mean_tokens * len(stability) * args.stability_repeats)
    report = {
        "model": args.model,
        "n": len(rows),
        "agreement": agree / len(rows),
        "heuristic_accepts": sum(1 for r in rows if r["heuristic"] == "accept"),
        "jev_accepts": sum(1 for r in rows if r["jev"] == "accept"),
        "disagreements": [r for r in rows if not r["agree"]],
        "stability": stability,
        "mean_stability_spread": (
            round(sum(s["spread"] for s in stability) / len(stability), 3)
            if stability else 0.0),
        "latency_ms": {"p50": lat[len(lat) // 2], "max": lat[-1]},
        "input_tokens": total_tokens,
        "est_cost_usd": round(total_tokens / 1e6 * PRICE_PER_MTOK, 6),
        "fallbacks": sum(1 for r in rows if r["source"] == "heuristic_fallback"),
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"{'pair':<22}{'heur':<8}{'jev':<8}{'agree':<7}{'p_acc':<7}{'lat_ms':<8}{'src'}")
    for r in rows:
        print(f"{r['name']:<22}{r['heuristic']:<8}{r['jev']:<8}"
              f"{str(r['agree']):<7}{r['p_accept']:<7.2f}"
              f"{r['latency_ms']:<8.0f}{r['source']}")
    print(f"\nagreement: {agree}/{len(rows)} "
          f"({report['agreement']:.0%}) | heuristic accepts "
          f"{report['heuristic_accepts']}, jev accepts {report['jev_accepts']}")
    print("stability (P(accept) across runs):")
    for s in stability:
        print(f"  {s['name']:<20} {s['p_accept_runs']} spread={s['spread']} flips={s['flips']}")
    print(f"latency p50 {report['latency_ms']['p50']:.0f}ms, max "
          f"{report['latency_ms']['max']:.0f}ms | ~{total_tokens} input tokens "
          f"~ ${report['est_cost_usd']:.6f}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
