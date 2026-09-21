"""ggt — the graphgen-toolgrad factory command line.

    python -m bridge.cli init --name acme --output acme.yaml
    python -m bridge.cli validate-config acme.yaml
    python -m bridge.cli run --config acme.yaml [--resume]
    python -m bridge.cli report --run-dir ./runs/acme
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_init(args) -> int:
    from bridge.config import write_example_config

    path = write_example_config(args.output, name=args.name)
    print(f"wrote scaffolded config: {path}")
    print("edit mcp_servers + domain, then: python -m bridge.cli validate-config " + path)
    return 0


def cmd_validate_config(args) -> int:
    from bridge.config import ConfigError, load_config

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 1
    print(f"config OK: {args.config}")
    print(f"  run name:   {cfg.name}")
    print(f"  servers:    {', '.join(s.name + '/' + s.transport for s in cfg.mcp_servers)}")
    print(f"  kg_source:  {cfg.domain.kg_source}")
    print(f"  chains:     {cfg.generation.num_chains} x "
          f"{cfg.generation.apis_per_workflow} apis")
    print(f"  llm:        {cfg.llm.backend}/{cfg.llm.model}")
    print(f"  output dir: {cfg.output.dir}")
    return 0


def cmd_run(args) -> int:
    from bridge.config import ConfigError, load_config
    from bridge.llm import LLMError, build_llm_client, credential_source_hint
    from bridge.runner import FactoryRun

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 1

    # The LLM client is optional: without a usable credential the run still
    # executes discovery/KG/ToolKG and generation, but refinement runs in
    # no-LLM mode (filtering only, no textual-gradient iterations).
    llm_client = None
    try:
        llm_client = build_llm_client(cfg.llm)
        print(f"llm credential: {credential_source_hint(cfg.llm)}")
    except LLMError as exc:
        print(f"note: {exc} — continuing without LLM refinement")
    if llm_client is None and cfg.domain.kg_source == "corpus":
        print("error: domain.kg_source='corpus' needs an LLM backend", file=sys.stderr)
        return 1

    run = FactoryRun(cfg, llm_client=llm_client)
    try:
        report = run.run(resume=args.resume)
    except Exception as exc:  # noqa: BLE001 — report already written by the runner
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"run {report['status']}: {os.path.join(run.workdir, 'run_report.json')}")
    return 0 if report["status"] == "ok" else 1


def cmd_report(args) -> int:
    path = os.path.join(args.run_dir, "run_report.json")
    if not os.path.exists(path):
        print(f"no run report at {path}", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    print(f"run: {report.get('run_name')}  status: {report.get('status')}  "
          f"duration: {report.get('duration_s')}s")
    for stage, summary in (report.get("stages") or {}).items():
        status = summary.get("status", "?")
        extra = summary.get("error") or ""
        print(f"  {stage:10s} {status:6s} {summary.get('duration_s', '?')}s {extra}")
    print(f"llm: {report.get('llm')}")
    print(f"eval: {report.get('eval')}")
    print(f"mix: {report.get('mix')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ggt", description="graphgen-toolgrad factory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="scaffold a config file")
    p_init.add_argument("--name", default="my-domain")
    p_init.add_argument("--output", default="ggt-config.yaml")
    p_init.set_defaults(func=cmd_init)

    p_validate = sub.add_parser("validate-config", help="check a config file")
    p_validate.add_argument("config")
    p_validate.set_defaults(func=cmd_validate_config)

    p_run = sub.add_parser("run", help="execute a factory run")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--resume", action="store_true",
                       help="skip stages with valid checkpoints")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="summarize a finished run")
    p_report.add_argument("--run-dir", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    sys.path.insert(0, _repo_root())
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
