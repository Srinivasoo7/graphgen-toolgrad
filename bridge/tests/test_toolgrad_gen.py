

def test_sanitize_proxy_env_strips_bracketed_ipv6():
    import os
    from bridge.toolgrad_gen import _sanitize_proxy_env

    old = os.environ.get("no_proxy")
    os.environ["no_proxy"] = "localhost,::1,[::1],example.com,[fd8b::1]"
    try:
        _sanitize_proxy_env()
        assert os.environ["no_proxy"] == "localhost,::1,example.com"
    finally:
        if old is None:
            del os.environ["no_proxy"]
        else:
            os.environ["no_proxy"] = old




def test_sanitize_proxy_env_noop_without_brackets():
    import os
    from bridge.toolgrad_gen import _sanitize_proxy_env

    old = os.environ.get("NO_PROXY")
    os.environ["NO_PROXY"] = "localhost,example.com"
    try:
        _sanitize_proxy_env()
        assert os.environ["NO_PROXY"] == "localhost,example.com"
    finally:
        if old is None:
            del os.environ["NO_PROXY"]
        else:
            os.environ["NO_PROXY"] = old


def _needs_networkx():
    try:
        import networkx  # noqa: F401
    except ImportError:
        return True
    return False


class _Patches:
    """Tiny monkeypatch stand-in for this fixture-free runner."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def setitem(self, mapping, key, value):
        self._undo.append(("item", mapping, key, key in mapping, mapping.get(key)))
        mapping[key] = value

    def undo(self):
        for entry in reversed(self._undo):
            if entry[0] == "item":
                _, mapping, key, had, old = entry
                if had:
                    mapping[key] = old
                else:
                    del mapping[key]
            else:
                obj, name, old = entry
                setattr(obj, name, old)


def _fake_fork(patches, *, invoke_impl, llms_per_chain=0):
    """Fake the ToolGrad fork pieces generate_samples touches (keyless).

    ``llms_per_chain``: how many times the fake graph calls the (real)
    ``create_llm`` factory per chain — simulates fork LLM usage so the
    ``max_llm_calls`` budget is tested through the real counter.
    """
    import sys
    import types
    from types import SimpleNamespace

    import bridge.toolgrad_gen as tg

    class FakeTracer:
        def __init__(self, output_dir=None, seed=None):
            pass

        def save(self):
            pass

    langchain_utils = SimpleNamespace()

    class FakeApp:
        def invoke(self, initial, config=None):
            for _ in range(llms_per_chain):
                langchain_utils.create_llm()
            return invoke_impl()

    prebuilt = SimpleNamespace(create_graph_on_mcp=lambda **kw: FakeApp())
    modules = SimpleNamespace(ToolGradState=lambda **kw: SimpleNamespace(**kw))
    mcp_module = SimpleNamespace()
    tog = SimpleNamespace(utils=SimpleNamespace(
        trace_utils=SimpleNamespace(ExecutionTracer=FakeTracer)))
    patches.setattr(tg, "_require_fork",
                    lambda: (tog, langchain_utils, mcp_module, prebuilt, modules))

    class FakeTool:
        def __init__(self, name):
            self.name = name
            self.description = f"{name} tool"
            self.args_schema = {"type": "object", "properties": {}}

    class FakeClient:
        def __init__(self, mcp_dict):
            pass

        async def get_tools(self):
            return [FakeTool("alpha"), FakeTool("beta")]

    fake_mod = types.ModuleType("langchain_mcp_adapters.client")
    fake_mod.MultiServerMCPClient = FakeClient
    patches.setitem(sys.modules, "langchain_mcp_adapters.client", fake_mod)

    # langchain_openai is only in the toolgrad venv; stub it so the real
    # _llm_factory (and its counter) runs keylessly too.
    openai_mod = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    openai_mod.ChatOpenAI = FakeChatOpenAI
    patches.setitem(sys.modules, "langchain_openai", openai_mod)


def _tmpdir():
    import tempfile
    return tempfile.TemporaryDirectory()


def test_generate_samples_records_per_chain_failures():
    if _needs_networkx():
        return "SKIP"
    import bridge.toolgrad_gen as tg

    tmp = _tmpdir()
    patches = _Patches()
    try:
        _fake_fork(patches, invoke_impl=lambda: {"workflow_cur": None})
        result = tg.generate_samples(
            mcp_dict={}, kg_text="kg", llm_cfg={}, num_chains=3,
            apis_per_workflow=2, seed=7, workdir=tmp.name, min_samples=0,
        )
    finally:
        patches.undo()
        tmp.cleanup()
    assert result["samples"] == []
    assert len(result["per_chain_failures"]) == 3
    assert all(f["error_type"] == "NoWorkflow"
               for f in result["per_chain_failures"])
    assert [f["seed"] for f in result["per_chain_failures"]] == [7, 8, 9]


def test_generate_samples_min_samples_enforced():
    if _needs_networkx():
        return "SKIP"
    import bridge.toolgrad_gen as tg

    tmp = _tmpdir()
    patches = _Patches()
    try:
        _fake_fork(patches, invoke_impl=lambda: {"workflow_cur": None})
        try:
            tg.generate_samples(
                mcp_dict={}, kg_text="kg", llm_cfg={}, num_chains=2,
                apis_per_workflow=2, seed=1, workdir=tmp.name,
            )
        except tg.GenerationError as exc:
            assert "min_samples=1" in str(exc)
            assert exc.details["num_samples"] == 0
            assert exc.details["min_samples"] == 1
            assert exc.details["num_chain_failures"] == 2
            assert len(exc.details["per_chain_failures"]) == 2
            assert "llm_calls" in exc.details
        else:
            raise AssertionError("expected GenerationError")
    finally:
        patches.undo()
        tmp.cleanup()


def test_generate_samples_chain_exception_recorded():
    if _needs_networkx():
        return "SKIP"
    import bridge.toolgrad_gen as tg

    def boom():
        raise RuntimeError("agent exploded")

    tmp = _tmpdir()
    patches = _Patches()
    try:
        _fake_fork(patches, invoke_impl=boom)
        result = tg.generate_samples(
            mcp_dict={}, kg_text="kg", llm_cfg={}, num_chains=2,
            apis_per_workflow=2, seed=1, workdir=tmp.name, min_samples=0,
        )
    finally:
        patches.undo()
        tmp.cleanup()
    assert len(result["per_chain_failures"]) == 2
    assert result["per_chain_failures"][0]["error_type"] == "RuntimeError"
    assert "agent exploded" in result["per_chain_failures"][0]["error"]


def test_generate_samples_max_llm_calls_stops_early():
    if _needs_networkx():
        return "SKIP"
    import bridge.toolgrad_gen as tg

    class Sample:
        def model_dump(self):
            return {"ok": True}

    tmp = _tmpdir()
    patches = _Patches()
    try:
        _fake_fork(patches, invoke_impl=lambda: {"workflow_cur": Sample()},
                   llms_per_chain=2)
        result = tg.generate_samples(
            mcp_dict={}, kg_text="kg", llm_cfg={}, num_chains=5,
            apis_per_workflow=2, seed=1, workdir=tmp.name,
            max_llm_calls=3,
        )
    finally:
        patches.undo()
        tmp.cleanup()
    # chain 0 builds 2 LLMs (2 <= 3, continues); chain 1 reaches 4 > 3: stop.
    assert result["llm_calls"] == 4
    assert len(result["samples"]) == 2
    assert result["per_chain_failures"] == []


def test_generate_samples_max_llm_calls_enforced_after_failures():
    # The soft cap is checked after every chain outcome, not just successes:
    # failing chains that burn LLM calls still trip the budget.
    if _needs_networkx():
        return "SKIP"
    import bridge.toolgrad_gen as tg

    def boom():
        raise RuntimeError("agent exploded")

    tmp = _tmpdir()
    patches = _Patches()
    try:
        _fake_fork(patches, invoke_impl=boom, llms_per_chain=2)
        result = tg.generate_samples(
            mcp_dict={}, kg_text="kg", llm_cfg={}, num_chains=5,
            apis_per_workflow=2, seed=1, workdir=tmp.name,
            max_llm_calls=3, min_samples=0,
        )
    finally:
        patches.undo()
        tmp.cleanup()
    # chain 0 fails after 2 calls (2 <= 3, continues); chain 1 fails after
    # 4 > 3: stop. Both failures recorded.
    assert result["llm_calls"] == 4
    assert len(result["per_chain_failures"]) == 2
    assert all(f["error_type"] == "RuntimeError"
               for f in result["per_chain_failures"])


def test_generate_samples_no_call_cap_runs_all_chains():
    if _needs_networkx():
        return "SKIP"
    import bridge.toolgrad_gen as tg

    class Sample:
        def model_dump(self):
            return {"ok": True}

    tmp = _tmpdir()
    patches = _Patches()
    try:
        _fake_fork(patches, invoke_impl=lambda: {"workflow_cur": Sample()},
                   llms_per_chain=2)
        result = tg.generate_samples(
            mcp_dict={}, kg_text="kg", llm_cfg={}, num_chains=4,
            apis_per_workflow=2, seed=1, workdir=tmp.name,
            max_llm_calls=0,
        )
    finally:
        patches.undo()
        tmp.cleanup()
    assert len(result["samples"]) == 4
    assert result["llm_calls"] == 8
