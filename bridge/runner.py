"""Staged, resumable factory runs: the product's main orchestration.

A :class:`FactoryRun` executes the full pipeline from a :class:`RunConfig`:

    discover (MCP) -> domain KG -> ToolKG -> generate (ToolGrad)
      -> refine/verify -> package (SFT mix) -> eval -> report

Properties an enterprise run needs:

- **checkpoints**: every stage writes ``checkpoints/<stage>.json``;
  ``run(resume=True)`` skips finished stages. Discovery always re-runs
  (live MCP sessions can't be pickled, and it's cheap); everything
  downstream resumes from disk. A checkpoint written by different bridge
  code is treated as absent, so code fixes are never silently skipped.
- **observability**: structured ``events.jsonl`` plus a human ``run.log``;
  per-stage timings in the final report.
- **spend control**: the LLM client's budget cap aborts before the next
  call; all LLM accounting lands in the report.
- **output safety**: PII redaction before packaging; a secret scan over
  every emitted file fails the run on any finding.
- **honesty**: stage failures are recorded with their exception and the
  run report is still written — a failed run explains itself.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from bridge import config as config_module
from bridge.config import RunConfig

log = logging.getLogger(__name__)

_STAGES = ["discover", "kg", "toolkg", "generate", "refine", "package", "eval"]

_BRIDGE_VERSION_CACHE: Optional[str] = None


def _bridge_version() -> str:
    """Content hash of the ``bridge`` package source (16 hex chars).

    Stamped into every stage checkpoint; on resume, a checkpoint written by
    different bridge code is treated as absent so code fixes can't be
    silently skipped by stale checkpoints. Cached per process.
    """
    global _BRIDGE_VERSION_CACHE
    if _BRIDGE_VERSION_CACHE is None:
        digest = hashlib.sha256()
        root = os.path.dirname(os.path.abspath(__file__))
        for name in sorted(os.listdir(root)):
            if name.endswith(".py"):
                with open(os.path.join(root, name), "rb") as fh:
                    digest.update(fh.read())
        _BRIDGE_VERSION_CACHE = digest.hexdigest()[:16]
    return _BRIDGE_VERSION_CACHE


class RunFailed(Exception):
    """A stage failed; the run report was still written."""


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class FactoryRun:
    """Execute the factory pipeline for ``config`` into ``workdir``."""

    def __init__(
        self,
        config: RunConfig,
        *,
        workdir: Optional[str] = None,
        llm_client=None,
        generate_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.workdir = os.path.abspath(workdir or config.output.dir)
        self.llm_client = llm_client
        self._generate_fn = generate_fn
        self._ctx: Dict[str, Any] = {}
        self._checkpoints_dir = os.path.join(self.workdir, "checkpoints")
        self._events_path = os.path.join(self.workdir, "events.jsonl")
        self._stage_summaries: Dict[str, Dict[str, Any]] = {}

    # -- plumbing ------------------------------------------------------
    def _event(self, stage: str, event: str, detail: Any = None) -> None:
        record = {"ts": _utcnow(), "stage": stage, "event": event}
        if detail is not None:
            record["detail"] = detail
        os.makedirs(os.path.dirname(self._events_path), exist_ok=True)
        with open(self._events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info("[%s] %s %s", stage, event, detail if detail is not None else "")

    def _checkpoint_path(self, stage: str) -> str:
        return os.path.join(self._checkpoints_dir, f"{stage}.json")

    def _save_checkpoint(self, stage: str, summary: Dict[str, Any]) -> None:
        summary["bridge_version"] = _bridge_version()
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        with open(self._checkpoint_path(stage), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    def _has_checkpoint(self, stage: str) -> bool:
        path = self._checkpoint_path(stage)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                stored = json.load(fh).get("bridge_version")
        except (OSError, ValueError):
            stored = None
        if stored != _bridge_version():
            # Code changed since this checkpoint was written (or it predates
            # versioning): re-run the stage rather than silently reusing
            # stale results.
            self._event(stage, "stale_checkpoint",
                        {"stored_bridge_version": stored,
                         "current_bridge_version": _bridge_version()})
            return False
        return True

    def _load_checkpoint(self, stage: str) -> Dict[str, Any]:
        with open(self._checkpoint_path(stage), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _run_stage(
        self,
        stage: str,
        fn: Callable[[], Dict[str, Any]],
        resume: bool,
        restore_fn: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        if resume and self._has_checkpoint(stage):
            summary = self._load_checkpoint(stage)
            if restore_fn is not None:
                restore_fn()
            self._event(stage, "resumed_from_checkpoint")
            self._stage_summaries[stage] = summary
            return summary
        self._event(stage, "started")
        started = time.monotonic()
        try:
            summary = fn()
        except Exception as exc:  # noqa: BLE001 — record, then propagate
            summary = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            details = getattr(exc, "details", None)
            if isinstance(details, dict) and details:
                summary["details"] = details
            summary["duration_s"] = round(time.monotonic() - started, 1)
            self._save_checkpoint(stage, summary)
            self._stage_summaries[stage] = summary
            self._event(stage, "failed", summary["error"])
            raise
        summary["status"] = "ok"
        summary["duration_s"] = round(time.monotonic() - started, 1)
        self._save_checkpoint(stage, summary)
        self._stage_summaries[stage] = summary
        self._event(stage, "finished", {k: v for k, v in summary.items() if k != "status"})
        return summary

    # -- stages ----------------------------------------------------------
    def _stage_discover(self) -> Dict[str, Any]:
        from bridge import mcp_client

        async def go():
            return await mcp_client.discover_all(self.config.mcp_servers)

        # Discovery runs on the run's home loop so the tools stay callable
        # for every later stage (refinement verification re-invokes them).
        discovered = self._ctx["loop_runner"].run(go())
        merged = mcp_client.merge_discovered(discovered)
        # Keep sessions alive for generate/refine; closed at end of run().
        self._ctx["discovered"] = merged
        tools_by_name = dict(merged.tools)
        self._ctx["tools_by_name"] = tools_by_name
        self._ctx["descriptors"] = list(merged.descriptors)
        by_server: Dict[str, int] = {}
        for d in merged.descriptors:
            by_server[d.server] = by_server.get(d.server, 0) + 1
        return {
            "num_tools": len(tools_by_name),
            "tools_by_server": by_server,
            "tool_names": sorted(tools_by_name),
        }

    def _stage_kg(self) -> Dict[str, Any]:
        from bridge import kg_builder, kg_context_exporter

        domain = self.config.domain
        llm_fn = self.llm_client.complete if self.llm_client else None
        spec = None
        if domain.kg_source == "spec" and domain.spec_file:
            spec = kg_builder.load_spec_file(domain.spec_file)
        graph = kg_builder.build_kg(
            domain.kg_source,
            descriptors=self._ctx.get("descriptors"),
            corpus_dir=domain.corpus_dir,
            spec=spec,
            llm_fn=llm_fn,
        )
        context = kg_context_exporter.export_kg_context(graph)
        kg_text = kg_context_exporter.render_kg_context(context)
        self._ctx["kg_context"] = context
        self._ctx["kg_text"] = kg_text
        path = os.path.join(self.workdir, "kg_context.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(context, fh, indent=2)
        return {
            "source": domain.kg_source,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "context_tokens_est": kg_context_exporter.estimate_tokens(kg_text),
            "path": path,
        }

    def _resume_kg(self) -> None:
        path = os.path.join(self.workdir, "kg_context.json")
        with open(path, "r", encoding="utf-8") as fh:
            context = json.load(fh)
        from bridge import kg_context_exporter

        self._ctx["kg_context"] = context
        self._ctx["kg_text"] = kg_context_exporter.render_kg_context(context)

    def _stage_toolkg(self) -> Dict[str, Any]:
        from bridge import toolkg_builder

        tools = list(self._ctx["tools_by_name"].values())
        hints = {
            name: [(p, t) for p, t in pairs]
            for name, pairs in self.config.generation.output_hints.items()
        }
        toolkg = toolkg_builder.build_toolkg(tools, output_hints=hints or None)
        self._ctx["toolkg"] = toolkg
        path = os.path.join(self.workdir, "toolkg.graphml")
        toolkg_builder.save_toolkg(toolkg, path)
        return {"stats": toolkg_builder.toolkg_stats(toolkg), "path": path}

    def _resume_toolkg(self) -> None:
        from bridge import toolkg_builder

        self._ctx["toolkg"] = toolkg_builder.load_toolkg(
            os.path.join(self.workdir, "toolkg.graphml")
        )

    def _stage_generate(self) -> Dict[str, Any]:
        if self._generate_fn is None:
            from bridge import toolgrad_gen

            generate_fn = toolgrad_gen.generate_samples
        else:
            generate_fn = self._generate_fn
        gen_cfg = self.config.generation
        result = generate_fn(
            mcp_dict=config_module.mcp_dict_for_config(self.config),
            kg_text=self._ctx["kg_text"],
            llm_cfg={
                "model": self.config.llm.model,
                "api_key_env": self.config.llm.api_key_env,
                "base_url": self.config.llm.base_url,
            },
            num_chains=gen_cfg.num_chains,
            apis_per_workflow=gen_cfg.apis_per_workflow,
            seed=gen_cfg.seed,
            min_samples=gen_cfg.min_samples,
            max_llm_calls=gen_cfg.max_llm_calls,
            output_hints={
                n: [(p, t) for p, t in pairs]
                for n, pairs in gen_cfg.output_hints.items()
            } or None,
            workdir=self.workdir,
        )
        # Tools from generation win: they are the wrapped objects the fork used.
        if result.get("tools_by_name"):
            self._ctx["tools_by_name"] = dict(result["tools_by_name"])
        self._ctx["samples"] = result["samples"]
        self._ctx["sample_paths"] = result["sample_paths"]
        self._ctx["trace_paths"] = result["trace_paths"]
        return {
            "num_samples": len(result["samples"]),
            "num_chains_requested": gen_cfg.num_chains,
            "num_chain_failures": len(result.get("per_chain_failures", [])),
            "per_chain_failures": result.get("per_chain_failures", []),
            "llm_calls": result.get("llm_calls", 0),
            "sample_paths": result["sample_paths"],
        }

    def _resume_generate(self) -> None:
        from bridge.trace_to_qa import load_workflow_sample

        summary = self._load_checkpoint("generate")
        paths = summary.get("sample_paths", [])
        self._ctx["samples"] = [load_workflow_sample(p) for p in paths]
        self._ctx["sample_paths"] = paths
        self._ctx["trace_paths"] = [
            os.path.join(self.workdir, "trace", f"{self.config.generation.seed + i:05d}.json")
            for i in range(len(paths))
        ]

    def _make_filter_fn(self, refine_config):
        quality = self.config.quality
        if quality.semantic_filter != "jev":
            return None
        from bridge.jev_filter import JevDecider

        decider = JevDecider(cli=quality.jev_cli, min_confidence=quality.jev_threshold)

        def jev_filter(qa: dict, score: dict) -> bool:
            decision = decider.decide(qa, score, refine_config)
            qa["jev"] = {
                "verdict": decision["verdict"],
                "confidence": decision["confidence"],
                "grounded": decision["grounded"],
                "latency_ms": decision["latency_ms"],
                "model": decision["model"],
                "source": decision["source"],
                "error": decision["error"],
            }
            return decision["verdict"] == "accept"

        return jev_filter

    def _stage_refine(self) -> Dict[str, Any]:
        from bridge import chain_verifier, refinement_loop, trace_to_qa
        from bridge.refinement_loop import RefineConfig, RefinementLoop, save_ledger

        quality = self.config.quality
        refine_config = RefineConfig(
            max_iterations=quality.max_iterations,
            min_coverage=quality.min_coverage,
            require_chain_valid=quality.require_chain_valid,
            require_entities=quality.require_entities,
            seed=self.config.generation.seed,
        )
        executor = chain_verifier.ToolGradExecutor(
            self._ctx["tools_by_name"],
            loop_runner=self._ctx.get("loop_runner"),
        )
        llm_fn = self.llm_client.complete if self.llm_client else None
        loop = RefinementLoop(
            llm_fn=llm_fn,
            kg_context=self._ctx["kg_context"],
            toolkg=self._ctx["toolkg"],
            executor=executor,
            config=refine_config,
            filter_fn=self._make_filter_fn(refine_config),
        )
        samples = self._ctx["samples"]
        tracers = [
            trace_to_qa.load_tracer(p)
            if p and os.path.exists(p)
            else None
            for p in self._ctx.get("trace_paths", [])
        ]
        tracers = (tracers + [None] * len(samples))[: len(samples)]
        kept, ledger = loop.run(samples, tracers)

        redaction_hits: Dict[str, int] = {}
        if quality.pii_redact:
            from bridge import redaction

            scrubbed = []
            for qa in kept:
                clean, hits = redaction.redact_qa(qa)
                scrubbed.append(clean)
                for label, n in hits.items():
                    redaction_hits[label] = redaction_hits.get(label, 0) + n
            kept = scrubbed
        self._ctx["kept"] = kept

        kept_path = os.path.join(self.workdir, "kept.json")
        with open(kept_path, "w", encoding="utf-8") as fh:
            json.dump(kept, fh, indent=2)
        save_ledger(ledger, os.path.join(self.workdir, "ledger.json"))
        last_iter = (ledger.get("iterations") or [{}])[-1]
        return {
            "stopped_reason": ledger.get("stopped_reason"),
            "num_kept": len(kept),
            "num_samples": len(samples),
            "num_chain_valid": last_iter.get("num_chain_valid"),
            "num_entity_grounded": last_iter.get("num_entity_grounded"),
            "mean_coverage": last_iter.get("mean_coverage"),
            "redaction_hits": redaction_hits,
            "kept_path": kept_path,
        }

    def _resume_refine(self) -> None:
        with open(os.path.join(self.workdir, "kept.json"), "r", encoding="utf-8") as fh:
            self._ctx["kept"] = json.load(fh)

    def _stage_package(self) -> Dict[str, Any]:
        from bridge import secret_scan, sft_mix
        from bridge.sft_mix import MixConfig

        kept = self._ctx["kept"]
        mix_dir = os.path.join(self.workdir, "sft_mix")
        os.makedirs(mix_dir, exist_ok=True)
        mix_config = MixConfig(
            valid_frac=1.0 - self.config.output.train_split,
            seed=self.config.generation.seed,
            require_chain_valid=self.config.quality.require_chain_valid,
            require_entities=self.config.quality.require_entities,
        )
        train, valid, mix_report = sft_mix.assemble(kept, config=mix_config)
        train_path = sft_mix.write_jsonl(train, os.path.join(mix_dir, "train.jsonl"))
        valid_path = sft_mix.write_jsonl(valid, os.path.join(mix_dir, "valid.jsonl"))
        self._ctx["mix_report"] = mix_report

        findings = secret_scan.scan_directory(self.workdir)
        clean = not findings
        summary: Dict[str, Any] = {
            "num_train": len(train),
            "num_valid": len(valid),
            "train_path": train_path,
            "valid_path": valid_path,
            "secret_scan_clean": clean,
            "secret_findings": [
                {"path": f.path, "line_no": f.line_no, "kind": f.kind}
                for f in findings
            ],
        }
        if not clean:
            raise RunFailed(
                "secret scan found credentials in run outputs: "
                + ", ".join(f"{f.kind}@{f.path}:{f.line_no}" for f in findings[:5])
            )
        return summary

    def _stage_eval(self) -> Dict[str, Any]:
        from bridge import chain_verifier, eval_harness

        executor = chain_verifier.ToolGradExecutor(
            self._ctx["tools_by_name"],
            loop_runner=self._ctx.get("loop_runner"),
        )
        metrics = eval_harness.evaluate(
            self._ctx["kept"],
            kg_context=self._ctx["kg_context"],
            toolkg=self._ctx["toolkg"],
            executor=executor,
        )
        self._ctx["eval_metrics"] = metrics
        return {"metrics": metrics}

    # -- public ----------------------------------------------------------
    def run(self, resume: bool = False) -> Dict[str, Any]:
        """Run all stages; always writes ``run_report.json`` at the end."""
        os.makedirs(self.workdir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.workdir, "run.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )
        self._event("run", "started", {"resume": resume, "workdir": self.workdir})
        run_started = time.monotonic()
        failed_stage: Optional[str] = None
        failed_exc: Optional[BaseException] = None
        # One event loop for the whole run: discovery and every later tool
        # call share it (see mcp_client.LoopRunner).
        from bridge import mcp_client as _mcp_client

        loop_runner = _mcp_client.LoopRunner()
        self._ctx["loop_runner"] = loop_runner
        try:
            # Discovery always re-runs: live MCP sessions can't be pickled.
            self._run_stage("discover", self._stage_discover, resume=False)
            if resume and self._has_checkpoint("kg"):
                self._run_stage("kg", self._resume_kg, resume=True,
                                restore_fn=self._resume_kg)
            else:
                self._run_stage("kg", self._stage_kg, resume=False)
            if resume and self._has_checkpoint("toolkg"):
                self._run_stage("toolkg", self._resume_toolkg, resume=True,
                                restore_fn=self._resume_toolkg)
            else:
                self._run_stage("toolkg", self._stage_toolkg, resume=False)
            if resume and self._has_checkpoint("generate"):
                self._run_stage("generate", self._resume_generate, resume=True,
                                restore_fn=self._resume_generate)
            else:
                self._run_stage("generate", self._stage_generate, resume=False)
            if resume and self._has_checkpoint("refine"):
                self._run_stage("refine", self._resume_refine, resume=True,
                                restore_fn=self._resume_refine)
            else:
                self._run_stage("refine", self._stage_refine, resume=False)
            self._run_stage("package", self._stage_package, resume=False)
            self._run_stage("eval", self._stage_eval, resume=False)
        except Exception as exc:  # noqa: BLE001 — report, then propagate
            failed_exc = exc
            for stage in _STAGES:
                summary = self._stage_summaries.get(stage)
                if summary is None or summary.get("status") != "ok":
                    failed_stage = stage
                    break
            self._event("run", "failed", f"{type(exc).__name__}: {exc}")
        finally:
            discovered = self._ctx.get("discovered")
            if discovered is not None:
                try:
                    loop_runner.run(discovered.aclose())
                except Exception:  # noqa: BLE001 — teardown must not fail the run
                    pass
            try:
                loop_runner.close()
            except Exception:  # noqa: BLE001 — teardown must not fail the run
                pass
        # The report is always written — a failed run explains itself.
        report = self._build_report(run_started, failed_stage)
        report_path = os.path.join(self.workdir, "run_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        self._event("run", "finished", {"report": report_path, "status": report["status"]})
        if failed_exc is not None:
            raise failed_exc
        return report

    def _build_report(self, run_started: float, failed_stage: Optional[str]) -> dict:
        llm_stats = self.llm_client.stats() if self.llm_client else {"calls": 0}
        eval_metrics = self._ctx.get("eval_metrics", {})
        mix_report = self._ctx.get("mix_report", {})
        return {
            "status": "failed" if failed_stage else "ok",
            "failed_stage": failed_stage,
            "run_name": self.config.name,
            "config_fingerprint": config_module.config_fingerprint(self.config),
            "dataset": {
                "name": self.config.output.dataset_name,
                "version": self.config.output.version,
            },
            "duration_s": round(time.monotonic() - run_started, 1),
            "stages": self._stage_summaries,
            "llm": llm_stats,
            "eval": {
                k: eval_metrics.get(k)
                for k in (
                    "fraction_chain_verified",
                    "mean_toolkg_coverage",
                    "entity_grounding_rate",
                )
            },
            "mix": {
                k: mix_report.get(k)
                for k in (
                    "n_bridge_in",
                    "n_bridge_after_cutoffs",
                    "n_bridge_dupes_removed",
                    "n_train",
                    "n_valid",
                )
            },
            "finished_at": _utcnow(),
        }
