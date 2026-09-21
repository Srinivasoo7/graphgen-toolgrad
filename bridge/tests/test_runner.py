"""End-to-end FactoryRun test with every external boundary faked (keyless).

Exercises: checkpointing, resume, events, PII redaction, secret scan,
run report, MCP discovery wiring, KG/tools->ToolKG->refine->package->eval.
"""

import json
import os
import tempfile

import networkx  # noqa: F401 — full dep set required; bare python3 SKIPs this module

import bridge.mcp_client as mcp_client
from bridge import runner as runner_module
from bridge.config import from_dict
from bridge.mcp_client import DiscoveredTools, ToolDescriptor
from bridge.runner import FactoryRun


class FakeTool:
    def __init__(self, name, description="", schema=None):
        self.name = name
        self.description = description
        self.args_schema = schema or {"type": "object", "properties": {}}
        self.coroutine = None

    def invoke(self, tool_input):
        return "ok"


def _fake_descriptors():
    return [
        ToolDescriptor(
            name="list_orders",
            description="List customer orders",
            input_schema={"properties": {"status": {"type": "string"}}},
            server="crm",
        ),
        ToolDescriptor(
            name="get_order",
            description="Get one order",
            input_schema={"properties": {"order_id": {"type": "string"}}},
            server="crm",
        ),
    ]


async def _fake_discover_all(servers, **kwargs):
    tools = [FakeTool("list_orders", "List customer orders"),
             FakeTool("get_order", "Get one order",
                      schema={"type": "object",
                              "properties": {"order_id": {"type": "string"}}})]
    return [DiscoveredTools(server_name="crm", tools={t.name: t for t in tools},
                            descriptors=_fake_descriptors(), _client=None)]


def _fake_generate(workdir, **kwargs):
    tools_by_name = {"list_orders": FakeTool("list_orders"),
                     "get_order": FakeTool("get_order")}
    sample = {
        "api_use_chains": {
            "0": {
                "description": "list then get",
                "intermediate_steps": [
                    ({"tool": "list_orders", "tool_input": {}},
                     "orders listed"),
                    ({"tool": "get_order", "tool_input": {"order_id": "1"}},
                     "order fetched"),
                ],
            }
        }
    }
    paths = []
    for i in range(2):
        path = os.path.join(workdir, f"sample-{i:05d}.json")
        with open(path, "w") as fh:
            json.dump(dict(sample), fh)
        paths.append(path)
    return {"samples": [sample, dict(sample)], "sample_paths": paths,
            "trace_paths": [], "tools_by_name": tools_by_name, "llm_calls": 0}


def _config(tmp):
    return from_dict({
        "name": "e2e-test",
        "mcp_servers": [{"name": "crm", "transport": "stdio",
                         "command": "npx", "args": ["-y", "x"]}],
        "domain": {"name": "crm", "kg_source": "tools"},
        "generation": {"num_chains": 2, "apis_per_workflow": 2, "seed": 1,
                       "output_hints": {"list_orders": [["order_id", "string"]]}},
        "llm": {"backend": "openrouter", "model": "m", "api_key_env": "X"},
        "quality": {"min_coverage": 0.5, "require_chain_valid": True,
                    "require_entities": False, "max_iterations": 2,
                    "pii_redact": True},
        "output": {"dir": os.path.join(tmp, "run"), "train_split": 0.5},
    })


def _patched():
    real = mcp_client.discover_all
    mcp_client.discover_all = _fake_discover_all
    return real


def test_full_run_ok():
    _tmp = tempfile.TemporaryDirectory()
    tmp_path = _tmp.name
    real = _patched()
    try:
        cfg = _config(str(tmp_path))
        run = FactoryRun(cfg, generate_fn=_fake_generate)
        report = run.run(resume=False)
    finally:
        mcp_client.discover_all = real
    assert report["status"] == "ok"
    assert report["failed_stage"] is None
    workdir = os.path.join(str(tmp_path), "run")
    for fname in ("run_report.json", "events.jsonl", "run.log",
                  "ledger.json", "kept.json", "kg_context.json",
                  "toolkg.graphml"):
        assert os.path.exists(os.path.join(workdir, fname)), fname
    train = os.path.join(workdir, "sft_mix", "train.jsonl")
    valid = os.path.join(workdir, "sft_mix", "valid.jsonl")
    assert os.path.exists(train) and os.path.exists(valid)
    stages = report["stages"]
    assert set(stages) == {"discover", "kg", "toolkg", "generate",
                           "refine", "package", "eval"}
    assert all(s["status"] == "ok" for s in stages.values())
    assert stages["refine"]["num_kept"] >= 1
    # gate breakdown is surfaced so empty results are diagnosable
    for key in ("num_chain_valid", "num_entity_grounded", "mean_coverage"):
        assert key in stages["refine"], key
    assert stages["package"]["secret_scan_clean"] is True
    assert report["config_fingerprint"]
    # events were logged
    with open(os.path.join(workdir, "events.jsonl")) as fh:
        events = [json.loads(line) for line in fh]
    assert any(e["event"] == "finished" and e["stage"] == "run" for e in events)


def test_resume_skips_finished_stages():
    _tmp = tempfile.TemporaryDirectory()
    tmp_path = _tmp.name
    real = _patched()
    try:
        cfg = _config(str(tmp_path))
        run = FactoryRun(cfg, generate_fn=_fake_generate)
        first = run.run(resume=False)
        # second run with resume: discovery re-runs, the rest hit checkpoints
        run2 = FactoryRun(cfg, generate_fn=_boom_generate)
        second = run2.run(resume=True)
    finally:
        mcp_client.discover_all = real
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    # generate_fn was never called on the resume path (would have raised)
    assert second["stages"]["refine"]["num_kept"] == first["stages"]["refine"]["num_kept"]


def _boom_generate(**kwargs):
    raise AssertionError("generate_fn must not run on the resume path")


def test_secret_scan_fails_run():
    _tmp = tempfile.TemporaryDirectory()
    tmp_path = _tmp.name
    real = _patched()
    try:
        cfg = _config(str(tmp_path))
        run = FactoryRun(cfg, generate_fn=_dirty_generate)
        try:
            run.run(resume=False)
        except Exception as exc:
            assert "secret scan" in str(exc)
        else:
            raise AssertionError("expected the secret scan to fail the run")
        # the failure is recorded in the report and checkpoint
        report = json.load(open(os.path.join(str(tmp_path), "run",
                                             "run_report.json")))
        assert report["status"] == "failed"
        assert report["failed_stage"] == "package"
        assert report["stages"]["package"]["status"] == "failed"
    finally:
        mcp_client.discover_all = real


def _dirty_generate(workdir, **kwargs):
    # plants a credential in the run outputs: the scan must catch it
    path = os.path.join(workdir, "leak.txt")
    with open(path, "w") as fh:
        fh.write('aws_key = "AKIAIOSFODNN7EXAMPLE"\n')
    return {"samples": [], "sample_paths": [], "trace_paths": [],
            "tools_by_name": {}, "llm_calls": 0}


def _empty_generate(workdir, **kwargs):
    # mirrors the real generate_samples min_samples enforcement
    from bridge import toolgrad_gen

    raise toolgrad_gen.GenerationError(
        "generation produced 0 sample(s), below min_samples=1 "
        "(2 chain failure(s) recorded; see per_chain_failures)",
        details={
            "num_samples": 0,
            "min_samples": 1,
            "llm_calls": 6,
            "num_chain_failures": 2,
            "per_chain_failures": [
                {"chain": 0, "seed": 1, "error_type": "NoWorkflow",
                 "error": "all iterations failed; no workflow produced"},
                {"chain": 1, "seed": 2, "error_type": "NoWorkflow",
                 "error": "all iterations failed; no workflow produced"},
            ],
        },
    )


def test_zero_samples_fails_run_with_report():
    _tmp = tempfile.TemporaryDirectory()
    tmp_path = _tmp.name
    real = _patched()
    try:
        cfg = _config(str(tmp_path))
        run = FactoryRun(cfg, generate_fn=_empty_generate)
        try:
            run.run(resume=False)
        except Exception as exc:
            assert "min_samples" in str(exc)
        else:
            raise AssertionError("expected zero samples to fail the run")
        report = json.load(open(os.path.join(str(tmp_path), "run",
                                             "run_report.json")))
        assert report["status"] == "failed"
        assert report["failed_stage"] == "generate"
        assert report["stages"]["generate"]["status"] == "failed"
        details = report["stages"]["generate"]["details"]
        assert details["num_samples"] == 0
        assert details["llm_calls"] == 6
        assert details["num_chain_failures"] == 2
        assert len(details["per_chain_failures"]) == 2
    finally:
        mcp_client.discover_all = real


def test_per_chain_failures_land_in_generate_summary():
    _tmp = tempfile.TemporaryDirectory()
    tmp_path = _tmp.name
    real = _patched()
    try:
        cfg = _config(str(tmp_path))

        def partial_generate(workdir, **kwargs):
            first = _fake_generate(workdir, **kwargs)
            first["per_chain_failures"] = [
                {"chain": 1, "seed": 2, "error_type": "NoWorkflow",
                 "error": "all iterations failed; no workflow produced"}
            ]
            return first

        run = FactoryRun(cfg, generate_fn=partial_generate)
        report = run.run(resume=False)
    finally:
        mcp_client.discover_all = real
    assert report["status"] == "ok"
    gen = report["stages"]["generate"]
    assert gen["num_chain_failures"] == 1
    assert gen["per_chain_failures"][0]["error_type"] == "NoWorkflow"


def test_discover_failure_fails_run():
    _tmp = tempfile.TemporaryDirectory()
    tmp_path = _tmp.name
    real = mcp_client.discover_all

    async def fail(servers, **kwargs):
        raise mcp_client.MCPError("nope")

    mcp_client.discover_all = fail
    try:
        cfg = _config(str(tmp_path))
        run = FactoryRun(cfg, generate_fn=_fake_generate)
        try:
            run.run(resume=False)
        except mcp_client.MCPError:
            pass
        else:
            raise AssertionError("expected MCPError")
        report = json.load(open(os.path.join(str(tmp_path), "run",
                                             "run_report.json")))
        assert report["status"] == "failed"
        assert report["failed_stage"] == "discover"
    finally:
        mcp_client.discover_all = real


def test_stale_checkpoint_reruns_stage():
    # A checkpoint written by different bridge code is treated as absent:
    # the stage re-runs instead of silently reusing stale results.
    _tmp = tempfile.TemporaryDirectory()
    tmp_path = _tmp.name
    real = _patched()
    try:
        cfg = _config(str(tmp_path))
        run = FactoryRun(cfg, generate_fn=_fake_generate)
        first = run.run(resume=False)
        assert first["status"] == "ok"
        ckpt = os.path.join(str(tmp_path), "run", "checkpoints", "generate.json")
        with open(ckpt) as fh:
            data = json.load(fh)
        assert data["bridge_version"]  # stamped on save
        data["bridge_version"] = "deadbeefdeadbeef"  # simulate a code change
        with open(ckpt, "w") as fh:
            json.dump(data, fh)
        calls = []

        def _recording_generate(**kwargs):
            calls.append(1)
            return _fake_generate(**kwargs)

        run2 = FactoryRun(cfg, generate_fn=_recording_generate)
        second = run2.run(resume=True)
    finally:
        mcp_client.discover_all = real
    assert second["status"] == "ok"
    assert calls, "stale generate checkpoint must trigger a re-run"
    with open(os.path.join(str(tmp_path), "run", "events.jsonl")) as fh:
        events = [json.loads(line) for line in fh]
    assert any(e["event"] == "stale_checkpoint" and e["stage"] == "generate"
               for e in events)
