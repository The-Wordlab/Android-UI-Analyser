"""Local adapter boundaries, without importing MLX or loading any model."""

import hashlib
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller import serve_local as local
from experiments.aua_controller import serve_model as offline


@pytest.fixture(autouse=True)
def reset_response_state():
    local._response_state.finish_reason = None
    local._response_state.envelope = None
    yield
    local._response_state.finish_reason = None
    local._response_state.envelope = None


@pytest.mark.parametrize("value", [
    'Example: call:tap_and_analyze{id:<|"|>el:fresh<|"|>}',
    'call:tap_and_analyze{id:<|"|>el:fresh<|"|>} extra',
    'call:tap_and_analyze{id:<|"|>el:fresh<|"|>} call:session_finish{',
    'call:session_finish{}call:session_finish{}',
    'call:tap_and_analyze{id:<|"|>el:fresh<|"|>',
    'call:tap_and_analyze{id:<|"|>el:fresh}',
    '<|tool_call>call:session_finish{}<tool_call|>',
])
def test_native_call_must_consume_the_entire_tool_channel(value):
    with pytest.raises(local.LocalProtocolError):
        local.single_native_call(value)


def test_nested_values_and_braces_inside_native_strings_are_preserved():
    value = 'call:input_and_analyze{id:<|"|>el:fresh<|"|>,text:<|"|>{"x": "雪"}\n}<|"|>}'
    assert local.single_native_call(value) == value


def test_formatter_returns_serialized_native_arguments_without_mutating_parser_output():
    parsed = {"name": "input_and_analyze", "arguments": {"id": "el:fresh", "text": '雪 {"a":1}'}}
    formatter = local.StrictToolCallFormatter(lambda *_: parsed, [])
    result = formatter(['call:input_and_analyze{}'])
    assert len(result) == 1 and result[0]["id"]
    assert result[0]["type"] == "function"
    assert json.loads(result[0]["function"]["arguments"]) == parsed["arguments"]
    assert isinstance(parsed["arguments"], dict)


def test_multiple_blocks_and_parser_errors_cannot_expose_a_valid_subset():
    calls = []
    def parser(*args):
        calls.append(args)
        raise ValueError("broken argument syntax")
    formatter = local.StrictToolCallFormatter(parser, [])
    with pytest.raises(local.LocalProtocolError, match="exactly one"):
        formatter(['call:session_finish{}', 'call:tap_and_analyze{'])
    assert not calls
    with pytest.raises(local.LocalProtocolError, match="could not be parsed"):
        formatter(['call:session_finish{}'])


def test_exhaustion_exposes_no_partial_call_and_preserves_length_state():
    local._response_state.finish_reason = "length"
    formatter = local.StrictToolCallFormatter(lambda *_: pytest.fail("parsed truncated output"), [])
    assert formatter(['call:tap_and_analyze{']) == []
    assert local._response_state.finish_reason == "length"


@pytest.mark.parametrize("change", [
    {"model": "google/unverified"}, {"model": "/tmp/model"}, {"adapters": "/tmp/adapter"},
    {"draft_model": "unverified"}, {"stream": True}, {"parallel_tool_calls": True},
    {"tool_choice": "required"}, {"chat_template_kwargs": {"enable_thinking": False}},
    {"chat_template_kwargs": {"enable_thinking": True, "tools": []}},
    {"max_tokens": 4097}, {"max_tokens": True}, {"max_tokens": 0},
    {"max_tokens": 4, "max_completion_tokens": 4}, {"temperature": True},
])
def test_fixed_local_request_rejects_model_and_lane_overrides(change):
    with pytest.raises(local.LocalProtocolError):
        local.validate_request({"messages": [{"role": "user", "content": "probe"}], **change})


def test_protocol_and_controller_request_shapes_are_accepted():
    for budget in [1024, 4096]:
        local.validate_request({"model": "default_model", "messages": [{"role": "user", "content": "probe"}],
                                "tools": [], "stream": False, "parallel_tool_calls": False,
                                "tool_choice": "auto", "max_tokens": budget, "temperature": 0,
                                "chat_template_kwargs": {"enable_thinking": True}})


def fake_server():
    class Provider:
        def load(self, *args):
            return args
    class Generator:
        def _tokenize(self, tokenizer, request, args):
            return request, [request], [], "reasoning"
        def generate(self, *args, **kwargs):
            context = SimpleNamespace(sequences={(1,): "<|tool_call>", (2,): "<tool_call|>"})
            tokens = kwargs.get("tokens", [SimpleNamespace(finish_reason=None, match=None),
                                           SimpleNamespace(finish_reason="length", match=None)])
            return context, iter(tokens)
    class Handler:
        def generate_response(self, text, finish_reason, *args, **kwargs):
            return {"content": text, "finish_reason": finish_reason, **kwargs}
    return SimpleNamespace(ModelProvider=Provider, ResponseGenerator=Generator, APIHandler=Handler)


def test_rendered_context_budget_includes_reserved_output_before_generation():
    _, generator_type, _ = local.guarded_classes(fake_server())
    generator = generator_type()
    args = SimpleNamespace(max_tokens=4096)
    assert len(generator._tokenize(None, [1] * (32768 - 4096), args)[0]) == 28672
    with pytest.raises(local.LocalProtocolError, match="rendered prompt"):
        generator._tokenize(None, [1] * (32768 - 4096 + 1), args)


def test_only_fixed_alias_can_reach_upstream_model_provider():
    provider_type, _, _ = local.guarded_classes(fake_server())
    provider = provider_type()
    assert provider.load("default_model") == ("default_model", None, "default_model")
    for parameters in [("unknown",), ("default_model", "/tmp/adapter"), ("default_model", None, "draft")]:
        with pytest.raises(local.LocalProtocolError):
            provider.load(*parameters)


def test_upstream_finish_reason_and_reasoning_survive_the_adapter():
    _, generator_type, handler_type = local.guarded_classes(fake_server())
    context, stream = generator_type().generate(None)
    assert context.sequences and len(list(stream)) == 2
    assert local._response_state.finish_reason == "length"
    handler = handler_type()
    result = handler.generate_response("", "length", reasoning_text="Unfinished thinking.", tool_calls=[])
    assert result["finish_reason"] == "length" and result["reasoning_text"] == "Unfinished thinking."
    with pytest.raises(local.LocalProtocolError, match="plain text"):
        handler.generate_response("call:session_finish{}", "stop", tool_calls=[])
    with pytest.raises(local.LocalProtocolError, match="visible text"):
        handler.generate_response("I might do this", "tool_calls", tool_calls=[{}])


@pytest.mark.parametrize("markers", [[(1,)], [(1,), (2,), (1,)], [(2,)], [(1,), (1,), (2,)]])
def test_balanced_arguments_without_one_closed_native_envelope_are_rejected(markers):
    _, generator_type, _ = local.guarded_classes(fake_server())
    tokens = [SimpleNamespace(match=m, finish_reason=None) for m in markers]
    tokens.append(SimpleNamespace(match=None, finish_reason="stop"))
    _, stream = generator_type().generate(tokens=tokens)
    list(stream)
    formatter = local.StrictToolCallFormatter(lambda *_: {"name": "session_finish", "arguments": {}}, [])
    with pytest.raises(local.LocalProtocolError, match="envelope"):
        formatter(['call:session_finish{}'])


def test_one_closed_native_envelope_can_return_the_exact_call():
    _, generator_type, _ = local.guarded_classes(fake_server())
    tokens = [SimpleNamespace(match=m, finish_reason=None) for m in [(1,), (2,)]]
    tokens.append(SimpleNamespace(match=None, finish_reason="stop"))
    _, stream = generator_type().generate(tokens=tokens)
    list(stream)
    formatter = local.StrictToolCallFormatter(lambda *_: {"name": "session_finish", "arguments": {}}, [])
    assert json.loads(formatter(['call:session_finish{}'])[0]["function"]["arguments"]) == {}


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    root.mkdir()
    files = {"config.json": b'{"model_type":"gemma4"}', "generation_config.json": b'{}',
             "tokenizer.json": b'{}', "tokenizer_config.json": b'{}',
             "chat_template.jinja": b'fictional template', "model.safetensors": b'fictional weights'}
    declarations = []
    for name, content in files.items():
        (root / name).write_bytes(content)
        declarations.append({"rfilename": name, "size": len(content),
                             "lfs": {"sha256": hashlib.sha256(content).hexdigest()}})
    monkeypatch.setitem(offline.TEMPLATE_HASHES, local.MODEL_ID,
                        hashlib.sha256(files["chat_template.jinja"]).hexdigest())
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"id": local.REPOSITORY, "sha": local.REVISION, "siblings": declarations}))
    monkeypatch.setattr(offline, "fetch", lambda *_: pytest.fail("network access"))
    return root, identity, tmp_path / "receipt.json"


def test_local_plan_verifies_pinned_snapshot_without_executing_cloud_plan(snapshot):
    root, identity, receipt = snapshot
    plan = local.local_plan(root, identity, receipt, 18000)
    assert plan["status"] == "verified_not_loaded"
    assert plan["served_model"] == "default_model" and plan["host"] == "127.0.0.1"
    assert plan["training_enabled"] is False and plan["offline"] is True
    assert plan["chat_template_kwargs"] == {"enable_thinking": True}
    assert "argv" not in plan and "shell_command" not in plan
    (root / "model.safetensors").write_bytes(b'wrong weights!!!')
    with pytest.raises(ValueError, match="snapshot artifact|Snapshot artifact"):
        local.local_plan(root, identity, receipt, 18000)


def test_local_plan_rejects_wrong_model_identity(snapshot):
    root, identity, receipt = snapshot
    data = json.loads(identity.read_text())
    data["sha"] = "a" * 40
    identity.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="repository and revision"):
        local.local_plan(root, identity, receipt, 18000)


def test_default_cli_only_verifies_and_records(snapshot, monkeypatch):
    root, identity, receipt = snapshot
    monkeypatch.setattr(sys, "argv", ["serve_local", "--snapshot-dir", str(root),
                                     "--identity-file", str(identity), "--receipt", str(receipt)])
    monkeypatch.setattr(local, "execute", lambda *_: pytest.fail("loaded model"))
    assert local.main() == 0
    assert json.loads(receipt.read_text())["status"] == "verified_not_loaded"


def test_execute_requires_apple_silicon_before_importing_mlx(monkeypatch, tmp_path):
    monkeypatch.setattr(local.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="Apple Silicon"):
        local.execute({}, tmp_path / "receipt.json")


def shared_model():
    return SimpleNamespace(
        args=SimpleNamespace(text_config={"num_hidden_layers": 42, "num_kv_shared_layers": 18}),
        layers=[SimpleNamespace(self_attn=SimpleNamespace(has_kv=i < 24)) for i in range(42)],
    )


def test_only_the_54_proven_unused_shared_kv_weights_are_skipped():
    extra = {"language_model.model.layers.23.self_attn.k_proj.weight": object(),
             "language_model.model.layers.24.self_attn.q_proj.weight": object(),
             "language_model.model.layers.24.self_attn.v_norm.weight": object(),
             "language_model.model.layers.24.self_attn.k_proj.bias": object(),
             "unexpected.weight": object()}
    weights = {**dict.fromkeys(local.SHARED_KV_UNUSED, object()), **extra}
    before = dict(weights)
    filtered, skipped = local.shared_kv_weights(shared_model(), weights)
    assert filtered == extra and len(skipped) == 54
    assert set(skipped) == local.SHARED_KV_UNUSED
    assert weights == before  # No input tensors or on-disk snapshot modified.


@pytest.mark.parametrize("mutation", ["active", "unknown", "module_present", "wrong_config"])
def test_shared_kv_filter_fails_when_unused_status_is_not_proven(mutation):
    model = shared_model()
    attention = model.layers[24].self_attn
    if mutation == "active":
        attention.has_kv = True
    elif mutation == "unknown":
        attention.has_kv = None
    elif mutation == "module_present":
        attention.k_proj = object()
    else:
        model.args.text_config["num_kv_shared_layers"] = 17
    with pytest.raises(ValueError, match="compatibility|proven unused"):
        local.shared_kv_weights(model, {"language_model.model.layers.24.self_attn.k_proj.weight": object()})


def test_sanitize_backport_retains_upstream_behavior_and_records_every_skip():
    class Model:
        def sanitize(self, weights):
            return {"language_model.model.layers.24.self_attn.k_proj.weight": weights["raw"],
                    "other": weights["other"]}
    model = Model()
    model.args, model.layers = shared_model().args, shared_model().layers
    plan = {}
    original = local.install_shared_kv_patch(Model, plan)
    try:
        result = model.sanitize({"raw": object(), "other": "retained"})
        assert result == {"other": "retained"}
        patch = plan["runtime_patches"][0]
        assert patch["strict_weight_loading"] is True
        assert patch["skipped_parameters"] == ["language_model.model.layers.24.self_attn.k_proj.weight"]
        assert patch["skipped_parameter_count"] == 1
        assert patch["upstream_fix"] == local.SHARED_KV_UPSTREAM
    finally:
        Model.sanitize = original


def threaded_server(events, *, load_error=None, native_tools=True, load_gate=None):
    """Model upstream's worker-owned startup without MLX arrays or any inference."""
    class Provider:
        def __init__(self, args=None):
            self.tokenizer = None

        def load_default(self):
            events.append(("load", threading.get_ident()))
            if load_gate is not None:
                assert load_gate.wait(2), "test did not release the fake load"
            if load_error is not None:
                raise load_error
            self.tokenizer = SimpleNamespace(has_tool_calling=native_tools, has_thinking=True)

    class Generator:
        def __init__(self, provider, cache=None):
            self.model_provider = provider
            self._stop_event = threading.Event()
            self._loop_entered = threading.Event()
            self._generation_thread = threading.Thread(target=self._generate)
            self._generation_thread.start()

        def _generate(self):
            self.model_provider.load_default()
            events.append(("generation_loop", threading.get_ident()))
            self._loop_entered.set()
            self._stop_event.wait()

        def stop_and_join(self):
            self._stop_event.set()
            self._generation_thread.join(timeout=2)
            assert not self._generation_thread.is_alive()

    return SimpleNamespace(ModelProvider=Provider, ResponseGenerator=Generator,
                           APIHandler=object, LRUPromptCache=lambda _: None)


def test_load_and_generation_remain_in_same_worker_and_readiness_waits_for_load():
    events, gate = [], threading.Event()
    provider_type, generator_type, _ = local.guarded_classes(threaded_server(events, load_gate=gate))
    provider = provider_type()
    assert events == []  # Constructing a provider must not preload on the caller.
    generator = generator_type(provider)
    try:
        assert not provider._load_ready.is_set()
        gate.set()
        generator.wait_until_ready()
        assert generator._loop_entered.wait(1)
        assert [name for name, _ in events] == ["load", "generation_loop"]
        assert len({worker for _, worker in events}) == 1
        assert events[0][1] != threading.get_ident()
    finally:
        gate.set()
        generator.stop_and_join()


@pytest.mark.parametrize("failure", ["weights", "tokenizer"])
def test_worker_load_or_native_tokenizer_failure_reaches_waiter(failure):
    error = ValueError("fixture strict load failure") if failure == "weights" else None
    server = threaded_server([], load_error=error, native_tools=failure != "tokenizer")
    provider_type, generator_type, _ = local.guarded_classes(server)
    generator = generator_type(provider_type())
    try:
        with pytest.raises(ValueError, match="fixture strict|native Gemma"):
            generator.wait_until_ready()
    finally:
        generator.stop_and_join()


def test_startup_timeout_does_not_claim_model_readiness():
    gate = threading.Event()
    provider_type, generator_type, _ = local.guarded_classes(threaded_server([], load_gate=gate))
    provider = provider_type()
    generator = generator_type(provider)
    try:
        with pytest.raises(TimeoutError, match="model readiness"):
            generator.wait_until_ready(timeout=0)
        assert not provider._load_ready.is_set()
    finally:
        gate.set()
        generator.stop_and_join()


@pytest.mark.parametrize("load_failure", [True, False])
def test_worker_startup_controls_http_and_receipt_and_restores_sanitize(snapshot, monkeypatch, load_failure):
    root, identity, receipt = snapshot
    plan = local.local_plan(root, identity, receipt, 18000)
    monkeypatch.setattr(local.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(local.importlib.metadata, "version", lambda name: local.RUNTIME[name])
    class Model:
        def sanitize(self, weights):
            return weights
    original = Model.sanitize
    events = []
    error = ValueError("fixture load failed, no real model accessed") if load_failure else None
    fake = threaded_server(events, load_error=error)
    server = ModuleType("mlx_lm.server")
    server.__dict__.update(vars(fake))
    server.__file__ = __file__
    def serve(host, port, generator, **kwargs):
        assert not load_failure
        assert json.loads(receipt.read_text())["status"] == "model_loaded"
        assert generator._loop_entered.wait(1)
        assert [name for name, _ in events] == ["load", "generation_loop"]
        assert events[0][1] != threading.get_ident()
        events.append(("http", threading.get_ident()))
        raise OSError("fixture bind failure, no socket opened")
    server._run_http_server = serve
    package = ModuleType("mlx_lm")
    package.server = server
    models = ModuleType("mlx_lm.models")
    models.gemma4 = SimpleNamespace(Model=Model, __file__=__file__)
    monkeypatch.setitem(sys.modules, "mlx_lm", package)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    expected_error = ValueError if load_failure else OSError
    with pytest.raises(expected_error, match="fixture load failed|fixture bind failure"):
        local.execute(plan, receipt)
    saved = json.loads(receipt.read_text())
    assert saved["status"] == ("load_failed" if load_failure else "serve_failed")
    assert saved["error_type"] == expected_error.__name__
    assert saved["runtime_patches"][0]["skipped_parameter_count"] == 0
    assert saved["model_load_execution"]["thread"] == "mlx_generation_worker"
    assert ("http" in [name for name, _ in events]) is not load_failure
    assert Model.sanitize is original
