"""Serve the verified Gemma E4B snapshot on this Mac through guarded MLX HTTP.

Inference only. The default verifies local hashes and writes a dry-run receipt;
--execute loads the model. No downloads, training, device or cloud operations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.aua_controller.serve_model import launch_plan, offline_plan

MODEL_ID = "gemma4-e4b"
REPOSITORY = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
ALIAS = "default_model"
HOST = "127.0.0.1"
MAX_CONTEXT = 32768
MAX_OUTPUT = 4096
RUNTIME = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}
_response_state = threading.local()
SHARED_KV_PATCH = "gemma4-e4b-shared-kv-unused-weights-v1"
SHARED_KV_UPSTREAM = "https://github.com/ml-explore/mlx-lm/commit/df1d3f3c9a7aae402dcbb8f41d4c36bcc13a50ae"
SHARED_KV_UNUSED = {
    f"language_model.model.layers.{layer}.self_attn.{projection}.weight"
    for layer in range(24, 42) for projection in ("k_norm", "k_proj", "v_proj")
}


class LocalProtocolError(ValueError):
    """Reject the entire request without exposing a partial native action."""


def shared_kv_weights(model: Any, weights: dict) -> tuple[dict, list[str]]:
    """Remove only the pinned E4B tensors unused by instantiated shared-KV layers.

    Narrow backport of upstream PR #1240 (SHARED_KV_UPSTREAM). Transformers also
    omits these modules and ignores their checkpoint entries.
    Everything else remains subject to MLX's unchanged strict weight loading.
    """
    config = model.args.text_config
    if (config.get("num_hidden_layers"), config.get("num_kv_shared_layers")) != (42, 18):
        raise ValueError("shared-KV compatibility requires the pinned 42-layer/18-shared E4B")
    result = dict(weights)
    skipped = []
    for name in sorted(SHARED_KV_UNUSED & weights.keys()):
        parts = name.split(".")
        attention = model.layers[int(parts[3])].self_attn
        if attention.has_kv is not False or hasattr(attention, parts[5]):
            raise ValueError("shared-KV tensor is not proven unused by the instantiated layer")
        del result[name]
        skipped.append(name)
    return result, skipped


def install_shared_kv_patch(model_class: type, plan: dict):
    original = model_class.sanitize
    patch = {"id": SHARED_KV_PATCH, "strict_weight_loading": True,
             "upstream_fix": SHARED_KV_UPSTREAM,
             "scope": "layers 24..41 with has_kv=False and absent k_norm/k_proj/v_proj modules",
             "skipped_parameters": [], "skipped_parameter_count": 0}
    plan["runtime_patches"] = [patch]

    def sanitize(model, weights):
        result, skipped = shared_kv_weights(model, original(model, weights))
        patch["skipped_parameters"] = skipped
        patch["skipped_parameter_count"] = len(skipped)
        return result

    model_class.sanitize = sanitize
    return original


def single_native_call(text: str) -> str:
    """Require complete consumption of one native call, respecting quoted braces."""
    if not isinstance(text, str):
        raise LocalProtocolError("native tool block must be text")
    value = text.strip()
    match = re.match(r"call:[A-Za-z_][A-Za-z0-9_-]*\{", value)
    if match is None:
        raise LocalProtocolError("expected one native call without surrounding prose")
    index = match.end() - 1
    depth = 0
    quoted = False
    while index < len(value):
        if value.startswith('<|"|>', index):
            quoted = not quoted
            index += 5
            continue
        if not quoted:
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    if index != len(value) - 1:
                        raise LocalProtocolError("extra or malformed content after native call")
                    return value
        index += 1
    raise LocalProtocolError("incomplete native call or quoted string")


class StrictToolCallFormatter:
    """MLX formatter seam: never salvage a valid subset of malformed tool output."""

    def __init__(self, tool_parser: Any, tools: Any, streaming: bool = False):
        if streaming:
            raise LocalProtocolError("streaming is unavailable in this local lane")
        self.parser, self.tools = tool_parser, tools

    def __call__(self, blocks: list[str]) -> list[dict]:
        # Preserve upstream exhaustion as length, including a partial final block.
        # No action is exposed; run_live rejects length before considering content.
        if getattr(_response_state, "finish_reason", None) == "length":
            return []
        if not blocks:
            return []
        if len(blocks) != 1:
            raise LocalProtocolError("exactly one native tool block is required")
        envelope = getattr(_response_state, "envelope", None)
        if envelope is not None and envelope != {"opened": 1, "closed": 1, "active": False, "invalid": False}:
            raise LocalProtocolError("native tool envelope was not opened and closed exactly once")
        text = single_native_call(blocks[0])
        try:
            call = self.parser(text, self.tools)
        except (ValueError, TypeError) as exc:
            raise LocalProtocolError("native tool arguments could not be parsed") from exc
        if (
            not isinstance(call, dict) or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), dict)
        ):
            raise LocalProtocolError("native parser must return one function with object arguments")
        return [{"id": "local-tool-" + uuid.uuid4().hex, "type": "function", "function": {
            "name": call["name"],
            "arguments": json.dumps(call["arguments"], ensure_ascii=False, allow_nan=False),
        }}]


def validate_request(body: dict) -> None:
    allowed = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream",
               "max_tokens", "max_completion_tokens", "temperature", "chat_template_kwargs"}
    if set(body) - allowed:
        raise LocalProtocolError("request contains options unavailable in this fixed local lane")
    if body.get("model", ALIAS) != ALIAS:
        raise LocalProtocolError("only the verified default_model alias is available")
    if body.get("stream", False) is not False or body.get("parallel_tool_calls", False) is not False:
        raise LocalProtocolError("only nonstreaming, single-call requests are supported")
    if body.get("tool_choice", "auto") != "auto":
        raise LocalProtocolError("tool_choice must be auto")
    if body.get("chat_template_kwargs", {"enable_thinking": True}) != {"enable_thinking": True}:
        raise LocalProtocolError("this lane requires enable_thinking=true without other overrides")
    temperature = body.get("temperature", 0)
    if type(temperature) not in {int, float} or temperature != 0:
        raise LocalProtocolError("this lane requires temperature zero")
    for key in ("max_tokens", "max_completion_tokens"):
        if key in body and (type(body[key]) is not int or not 1 <= body[key] <= MAX_OUTPUT):
            raise LocalProtocolError("requested output must be between 1 and 4096 tokens")
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise LocalProtocolError("use one output token budget field")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise LocalProtocolError("nonempty chat messages are required")


def guarded_classes(server: Any) -> tuple[type, type, type]:
    """Small, version-pinned seams around upstream loading, tokenization and HTTP."""
    class FixedModelProvider(server.ModelProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._load_ready = threading.Event()
            self._load_error = None

        def load_default(self):
            # Upstream PR #1090 deliberately loads inside the generation worker:
            # lazy non-parameter tensors can retain their creating thread's stream.
            super().load_default()
            if not self.tokenizer.has_tool_calling or not self.tokenizer.has_thinking:
                raise ValueError("verified tokenizer did not enable native Gemma tools and thinking")
            self._load_ready.set()

        def load(self, model_path, adapter_path=None, draft_model_path=None):
            if model_path != ALIAS or adapter_path is not None or draft_model_path not in {None, ALIAS}:
                raise LocalProtocolError("model, adapter and draft overrides are unavailable")
            return super().load(ALIAS, None, ALIAS)

    class BoundedGenerator(server.ResponseGenerator):
        def _generate(self):
            try:
                # Keep upstream's default-stream selection followed by model load.
                super()._generate()
            except Exception as exc:
                if self.model_provider._load_ready.is_set():
                    raise
                self.model_provider._load_error = exc
                self.model_provider._load_ready.set()

        def wait_until_ready(self, timeout=120.0):
            provider = self.model_provider
            deadline = time.monotonic() + timeout
            while not provider._load_ready.wait(min(0.1, max(0, deadline - time.monotonic()))):
                if not self._generation_thread.is_alive():
                    raise RuntimeError("MLX generation worker stopped before model readiness")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"MLX worker model readiness exceeded {timeout:g} seconds")
            if provider._load_error is not None:
                raise provider._load_error
            if not self._generation_thread.is_alive():
                raise RuntimeError("MLX generation worker stopped before HTTP startup")

        def _tokenize(self, tokenizer, request, args):
            result = super()._tokenize(tokenizer, request, args)
            if len(result[0]) + args.max_tokens > MAX_CONTEXT:
                raise LocalProtocolError("rendered prompt plus requested output exceeds 32768 tokens")
            return result

        def generate(self, *args, **kwargs):
            _response_state.finish_reason = None
            envelope = {"opened": 0, "closed": 0, "active": False, "invalid": False}
            _response_state.envelope = envelope
            context, output = super().generate(*args, **kwargs)

            def tracked():
                for token in output:
                    marker = context.sequences.get(token.match) if token.match is not None else None
                    if marker == "<|tool_call>":
                        envelope["invalid"] |= envelope["active"]
                        envelope["opened"] += 1
                        envelope["active"] = True
                    elif marker == "<tool_call|>":
                        envelope["invalid"] |= not envelope["active"]
                        envelope["closed"] += 1
                        envelope["active"] = False
                    if token.finish_reason is not None:
                        _response_state.finish_reason = token.finish_reason
                    yield token
            return context, tracked()

    class LocalHandler(server.APIHandler):
        def _local_error(self, message: str, status: int = 400):
            # Nonstreaming upstream buffers headers until parsing has completed.
            self._headers_buffer = []
            self._set_completion_headers(status)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {
                "code": "local_protocol_error", "message": message,
            }}).encode())

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._local_error("only /v1/chat/completions is supported", 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_048_576:
                    raise LocalProtocolError("request body exceeds the local limit or is empty")
                super().do_POST()
            except LocalProtocolError as exc:
                self._local_error(str(exc), 422)
            except (ValueError, TypeError, KeyError, AssertionError):
                self._local_error("invalid local chat request or native response", 422)

        def handle_completion(self, request, stop_words):
            validate_request(self.body)
            return super().handle_completion(request, stop_words)

        def generate_response(self, text, finish_reason, *args, **kwargs):
            calls = kwargs.get("tool_calls") or []
            if finish_reason != "length":
                if calls and text.strip():
                    raise LocalProtocolError("native call accompanied by unparsed visible text")
                if not calls and (finish_reason == "tool_calls" or "call:" in text
                                  or "<|tool_call>" in text):
                    raise LocalProtocolError("native tool output must not fall back to plain text")
                if not calls and (getattr(_response_state, "envelope", None) or {}).get("opened"):
                    raise LocalProtocolError("native tool envelope produced no complete call")
            return super().generate_response(text, finish_reason, *args, **kwargs)

        def handle_models_request(self):
            self._set_completion_headers(200)
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [{
                "id": ALIAS, "object": "model", "owned_by": "local",
            }]}).encode())

    return FixedModelProvider, BoundedGenerator, LocalHandler


def local_plan(snapshot: Path, identity: Path, receipt: Path, port: int) -> dict:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    manifest = json.loads(Path(__file__).with_name("comparison.json").read_text())
    model = next(item for item in manifest["models"] if item["id"] == MODEL_ID)
    if (model["repository"], model["revision"], model["model_type"]) != (REPOSITORY, REVISION, "gemma4"):
        raise ValueError("local lane requires the fixed official Gemma E4B revision")
    verified = offline_plan(launch_plan(model, receipt.parent, False, max_model_len=MAX_CONTEXT),
                            snapshot, identity)
    return {
        "format": "aua-controller-local-mlx-v1", "status": "verified_not_loaded",
        "created_utc": datetime.now(UTC).isoformat(), "execution_location": "local_mac",
        "training_enabled": False, "offline": True, "model": model,
        "served_model": ALIAS, "host": HOST, "port": port,
        "snapshot_dir": verified["snapshot_dir"], "identity_file": verified["identity_file"],
        "hub_declared_files": verified["hub_declared_files"],
        "chat_template_sha256": verified["chat_template_sha256"],
        "chat_template_kwargs": {"enable_thinking": True}, "precision": "bfloat16",
        "max_context_tokens": MAX_CONTEXT, "max_output_tokens": MAX_OUTPUT,
        "context_limit_enforced": "full rendered prompt plus requested output before generation",
        "concurrency": 1, "temperature": 0, "expected_runtime": RUNTIME,
        "model_load_execution": {
            "thread": "mlx_generation_worker",
            "upstream_reference": "https://github.com/ml-explore/mlx-lm/pull/1090",
            "http_startup": "after worker model load and native tokenizer validation",
            "readiness_timeout_seconds": 120,
        },
        "reasoning_tokens": "not separately reported by upstream MLX; do not assume zero",
        "limitations": ["upstream cancellation may not interrupt a running prefill immediately",
                        "readiness timeout prevents HTTP startup; joining an active model load may take longer",
                        "runtime identity and native tool support do not establish AUA task success"],
    }


def execute(plan: dict, receipt: Path) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("execution requires an Apple Silicon Mac")
    actual = {package: importlib.metadata.version(package) for package in RUNTIME}
    if actual != RUNTIME:
        raise ValueError("local MLX package versions differ from the inspected runtime")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    plan["runtime_versions"] = actual
    plan["status"] = "loading"
    receipt.write_text(json.dumps(plan, indent=2) + "\n")
    generator = None
    patched_class = original_sanitize = None
    loaded = False
    try:
        from mlx_lm import server
        from mlx_lm.models import gemma4

        provider_type, generator_type, handler_type = guarded_classes(server)
        server.ToolCallFormatter = StrictToolCallFormatter
        args = SimpleNamespace(
            model=plan["snapshot_dir"], adapter_path=None, draft_model=None, pipeline=False,
            trust_remote_code=False, use_default_chat_template=False,
            chat_template=(Path(plan["snapshot_dir"]) / "chat_template.jinja").read_text(),
            chat_template_args={"enable_thinking": True}, allowed_origins=[],
            max_tokens=MAX_OUTPUT, temp=0.0, top_p=1.0, top_k=0, min_p=0.0,
            num_draft_tokens=0, decode_concurrency=1, prompt_concurrency=1,
            prefill_step_size=512, prompt_cache_size=1, prompt_cache_bytes=None,
        )
        plan["runtime_source_sha256"] = {
            "mlx_lm.server": hashlib.sha256(Path(server.__file__).read_bytes()).hexdigest(),
            "mlx_lm.models.gemma4": hashlib.sha256(Path(gemma4.__file__).read_bytes()).hexdigest(),
            "local_launcher": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        patched_class = gemma4.Model
        original_sanitize = install_shared_kv_patch(patched_class, plan)
        provider = provider_type(args)
        generator = generator_type(provider, server.LRUPromptCache(1))
        generator.wait_until_ready()
        loaded = True
        plan["status"] = "model_loaded"
        receipt.write_text(json.dumps(plan, indent=2) + "\n")
        server._run_http_server(HOST, plan["port"], generator,
                                server_class=HTTPServer, handler_class=handler_type)
    except Exception as exc:
        plan["status"] = "serve_failed" if loaded else "load_failed"
        plan["error_type"] = type(exc).__name__
        raise
    finally:
        # Upstream also joins on KeyboardInterrupt. Thread.join is repeatable,
        # but skip the second call after its orderly shutdown.
        if generator is not None and generator._generation_thread.is_alive():
            generator.stop_and_join()
        if patched_class is not None and original_sanitize is not None:
            patched_class.sanitize = original_sanitize
        if plan["status"] not in {"load_failed", "serve_failed"}:
            plan["status"] = "stopped"
        receipt.write_text(json.dumps(plan, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--identity-file", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--port", default=18000, type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = local_plan(args.snapshot_dir, args.identity_file, args.receipt, args.port)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"status": plan["status"], "receipt": str(args.receipt),
                      "served_model": ALIAS, "host": HOST, "port": args.port}), flush=True)
    if args.execute:
        execute(plan, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
