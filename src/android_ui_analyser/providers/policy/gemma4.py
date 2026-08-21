"""Optional local Gemma 4 semantic reviewer for guarded AUA policy candidates.

This provider is text-only: it receives the same privacy-screened candidate projection as
FunctionGemma and never receives screenshots, hierarchy dumps, typed values, or trusted call
arguments. It loads only an existing absolute MLX model directory and never downloads weights.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import json
import platform
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...model_control import model_context_window
from ...policy import POLICY_HANDOFF_ID, PolicyContext, compile_policy_context, guard_candidates
from ..base import Availability, PolicyProvider
from ..registry import register_policy

#: The pip extra that installs this reviewer's inference runtime (mlx-vlm).
INSTALL_EXTRA = "hybrid-policy"
_FINAL_MARKER = "FINAL_CANDIDATE_ID="
# The contract is "exactly one verdict, and nothing after it" — not a particular reasoning
# format. Anchoring on the thinking channel's exact delimiters made a well-formed verdict
# unparseable whenever the model varied or omitted that wrapper, which silently removed the
# reviewer from the chain. Reasoning may take any shape; the verdict must still end the output.
_FINAL_CANDIDATE = re.compile(rf".*?{_FINAL_MARKER}(-?[0-9]+)\s*\Z", re.DOTALL)
_VERDICT_TAIL = re.compile(r"-?[0-9]+\s*\Z")
_SYSTEM_PROMPT = (
    "You are a careful Android UI navigation critic. AUA already screened every supplied "
    "current-frame control for safety and authored the exact calls. Think about the earliest "
    "incomplete destination in the goal and compare the semantic meaning of every control. "
    "Candidate IDs and order are arbitrary. Choose only the next action toward the earliest "
    "incomplete waypoint; later steps do not need to be visible on the current screen. Do not "
    "prefer a prominent content card. If exactly one candidate directly advances that next "
    "waypoint, select it. If none does, select -1. Keep reasoning concise. After reasoning, "
    "output exactly FINAL_CANDIDATE_ID=<integer> and nothing after it."
)


def _absolute_model_directory(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("models.gemma4.model_path must be an absolute local directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("models.gemma4.model_path must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("models.gemma4.model_path must be a directory")
    if not (resolved / "config.json").is_file():
        raise ValueError("Gemma 4 model directory is missing config.json")
    if not any(resolved.glob("*.safetensors")):
        raise ValueError("Gemma 4 model directory contains no safetensors weights")
    return resolved


def verdict_failure_kind(output: str) -> str:
    """Classify an unparseable verdict without echoing any model or app text."""

    count = output.count(_FINAL_MARKER)
    if count == 0:
        return "no_verdict_marker"
    if count > 1:
        return "multiple_verdict_markers"
    if not _VERDICT_TAIL.fullmatch(output.rsplit(_FINAL_MARKER, 1)[1]):
        return "malformed_or_trailing_verdict"
    return "unknown"


def parse_candidate_id(output: str) -> int:
    """Parse one exact reviewer verdict and reject prose or trailing output."""

    if output.count(_FINAL_MARKER) != 1:
        raise ValueError("Gemma 4 output is not one exact candidate verdict")
    match = _FINAL_CANDIDATE.fullmatch(output)
    if match is None:
        raise ValueError("Gemma 4 output is not one exact candidate verdict")
    return int(match.group(1))


@register_policy("gemma4")
class Gemma4PolicySelector(PolicyProvider):
    """Lazy, resident MLX Gemma 4 reviewer over already-guarded opaque candidates."""

    def __init__(
        self,
        settings: Mapping[str, Any] | None = None,
        *,
        model_loader: Callable[..., Any] | None = None,
        generator: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(settings)
        self._model_loader = model_loader
        self._generator = generator
        self._model: Any | None = None
        self._processor: Any | None = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self.last_error: str | None = None
        self.last_selection: dict[str, Any] = {}

    def supports_candidate_count(self, count: int) -> bool:
        return 2 <= count <= 4

    def supports_handoff(self) -> bool:
        return True

    def supports_mode(self, mode: str) -> bool:
        if mode == "shadow":
            return True
        return mode == "advisory" and self.settings.get("max_mode") == "advisory"

    def _max_tokens(self) -> int:
        value = self.settings.get("max_tokens", 512)
        if not isinstance(value, int) or isinstance(value, bool) or not 32 <= value <= 1024:
            raise ValueError("models.gemma4.max_tokens must be an integer from 32 to 1024")
        return value

    def _runtime_availability(self) -> Availability:
        if self._model_loader is not None and self._generator is not None:
            return Availability(True, "injected runtime")
        if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            return Availability(False, "Gemma 4 MLX policy requires Apple silicon")
        if importlib.util.find_spec("mlx_vlm") is None:
            return Availability(
                False,
                f"optional dependency missing; install android-ui-analyser[{INSTALL_EXTRA}]",
            )
        return Availability(True, "local MLX-VLM runtime available")

    def is_available(self) -> Availability:
        try:
            runtime = self._runtime_availability()
            if not runtime.ok:
                return runtime
            _absolute_model_directory(self.settings.get("model_path"))
            self._max_tokens()
        except Exception as exc:
            self.last_error = str(exc)
            return Availability(False, str(exc))
        self.last_error = None
        return Availability(True, "local Gemma 4 reviewer is ready")

    def status(self) -> dict[str, Any]:
        available = self.is_available()
        try:
            runtime = self._runtime_availability()
        except Exception:  # pragma: no cover - defensive
            runtime = Availability(False, "policy runtime availability check failed")
        return {
            "provider": self.name,
            "available": available.ok,
            "reason": available.reason,
            "loaded": self._model is not None,
            "context_window": model_context_window(self.settings),
            "supported_candidate_counts": [2, 3, 4],
            "supports_handoff": True,
            # Reported apart from `available` so a caller can tell "the runtime is not installed
            # in this environment" (one fix) from "the base model is not configured" (another).
            "install_extra": INSTALL_EXTRA,
            "runtime": {"ready": runtime.ok, "reason": runtime.reason},
            "rollout": {
                "max_mode": self.settings.get("max_mode", "shadow"),
                "supported_modes": [
                    "shadow",
                    *(["advisory"] if self.settings.get("max_mode") == "advisory" else []),
                ],
                "source": "explicit_local_operator_config",
            },
            "artifacts": {
                "model_path": self.settings.get("model_path"),
                "revision": self.settings.get("revision"),
            },
            "last_error": self.last_error,
            "last_selection": dict(self.last_selection),
        }

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor
            model_path = _absolute_model_directory(self.settings.get("model_path"))
            if self._model_loader is None or self._generator is None:
                from mlx_vlm import generate, load

                self._model_loader = self._model_loader or load
                self._generator = self._generator or generate
            loader = self._model_loader
            if loader is None:
                raise RuntimeError("Gemma 4 model loader is unavailable")
            loaded = loader(str(model_path))
            if not isinstance(loaded, tuple) or len(loaded) < 2:
                raise RuntimeError("Gemma 4 model loader returned an invalid result")
            self._model, self._processor = loaded[:2]
        return self._model, self._processor

    def load_model(self) -> dict[str, Any]:
        operation_id = uuid.uuid4().hex
        started = time.perf_counter()
        self.emit_model_event({"id": operation_id, "source": "dashboard", "phase": "loading"})
        try:
            self._ensure_loaded()
        except Exception as exc:
            self.emit_model_event(
                {
                    "id": operation_id,
                    "source": "dashboard",
                    "phase": "error",
                    "operation": "load",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        duration = round((time.perf_counter() - started) * 1000, 1)
        self.emit_model_event(
            {
                "id": operation_id,
                "source": "dashboard",
                "phase": "complete",
                "operation": "load",
                "duration_ms": duration,
            }
        )
        return {"ok": True, "provider": self.name, "loaded": True, "duration_ms": duration}

    def unload_model(self) -> dict[str, Any]:
        with self._load_lock:
            self._model = None
            self._processor = None
        gc.collect()
        with contextlib.suppress(Exception):
            import mlx.core as mx

            mx.clear_cache()
        self.emit_model_event({"source": "dashboard", "phase": "complete", "operation": "unload"})
        return {"ok": True, "provider": self.name, "loaded": False}

    @staticmethod
    def _token_count(processor: Any, text: str) -> int | None:
        tokenizer = getattr(processor, "tokenizer", processor)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            return None
        try:
            return len(encode(text))
        except Exception:
            return None

    def interact(
        self, messages: list[dict[str, str]], *, max_tokens: int | None = None
    ) -> dict[str, Any]:
        operation_id = uuid.uuid4().hex
        budget = max(1, min(int(max_tokens or self._max_tokens()), 1024))
        started = time.perf_counter()
        self.emit_model_event(
            {
                "id": operation_id,
                "source": "playground",
                "phase": "running",
                "input": messages,
                "max_tokens": budget,
                "context_window": model_context_window(self.settings),
            }
        )
        try:
            model, processor = self._ensure_loaded()
            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            generator = self._generator
            if generator is None:
                raise RuntimeError("Gemma 4 generator is unavailable")
            seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
            with self._generation_lock:
                generated = generator(
                    model,
                    processor,
                    prompt,
                    max_tokens=budget,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    enable_thinking=True,
                    verbose=False,
                    seed=seed,
                )
            output = str(getattr(generated, "text", generated))
            output_tokens = getattr(generated, "generation_tokens", None)
            input_tokens = self._token_count(processor, prompt)
        except Exception as exc:
            self.emit_model_event(
                {
                    "id": operation_id,
                    "source": "playground",
                    "phase": "error",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        result = {
            "ok": True,
            "provider": self.name,
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_tokens": budget,
            "context_window": model_context_window(self.settings),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        self.emit_model_event(
            {"id": operation_id, "source": "playground", "phase": "complete", **result}
        )
        return result

    @staticmethod
    def _review_state(context: PolicyContext) -> dict[str, Any]:
        compiled = compile_policy_context(context)
        return {
            "goal": context.goal,
            "phase": context.phase,
            "observation": compiled["observation"],
            "constraints": [
                "Choose only a supplied current-frame control.",
                "Do not execute or rewrite the action.",
                "Return -1 rather than guessing.",
            ],
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "call": candidate.as_model_value()["call"],
                    "purpose": candidate.purpose,
                    "proof": candidate.proof,
                }
                for candidate in context.candidates
            ],
        }

    def select(self, context: PolicyContext) -> int | None:
        operation_id = uuid.uuid4().hex
        started = time.perf_counter()
        messages: list[dict[str, str]] | None = None
        output: str | None = None
        try:
            guarded = guard_candidates(
                context,
                max_candidates=max(1, min(4, len(context.candidates))),
            )
            if guarded != context.candidates:
                raise ValueError("provider received candidates that did not pass the policy guard")
            if not self.supports_candidate_count(len(guarded)):
                raise ValueError("unsupported candidate cardinality")
            model, processor = self._ensure_loaded()
            state = self._review_state(context)
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        state,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            ]
            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            self.emit_model_event(
                {
                    "id": operation_id,
                    "source": "agent",
                    "phase": "running",
                    "input": messages,
                    "input_tokens": self._token_count(processor, prompt),
                    "max_tokens": self._max_tokens(),
                    "context_window": model_context_window(self.settings),
                }
            )
            seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
            generator = self._generator
            if generator is None:
                raise RuntimeError("Gemma 4 generator is unavailable")
            with self._generation_lock:
                result = generator(
                    model,
                    processor,
                    prompt,
                    max_tokens=self._max_tokens(),
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    enable_thinking=True,
                    verbose=False,
                    seed=seed,
                )
            output = str(getattr(result, "text", result))
            generated = getattr(result, "generation_tokens", None)
            try:
                selected_id = parse_candidate_id(output)
            except ValueError as exc:
                # An unusable verdict silently removes the reviewer from the chain, so record
                # why. These counters describe the *shape* of the output only: no model text,
                # candidate label, or resource id is retained.
                budget = self._max_tokens()
                self.last_selection = {
                    "parsed": False,
                    "failure": verdict_failure_kind(output),
                    "generation_tokens": generated,
                    "max_tokens": budget,
                    "truncated": isinstance(generated, int) and generated >= budget,
                    "output_chars": len(output),
                    "verdict_markers": output.count(_FINAL_MARKER),
                }
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.emit_model_event(
                    {
                        "id": operation_id,
                        "source": "agent",
                        "phase": "error",
                        "input": messages,
                        "output": output,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        "error": self.last_error,
                        **self.last_selection,
                    }
                )
                return None
            offered = {candidate.candidate_id for candidate in guarded}
            if selected_id != POLICY_HANDOFF_ID and selected_id not in offered:
                raise ValueError("Gemma 4 selected an ID outside the guarded candidates")
            self.last_selection = {
                "parsed": True,
                "selected_id": selected_id,
                "generation_tokens": generated,
            }
            self.last_error = None
            self.emit_model_event(
                {
                    "id": operation_id,
                    "source": "agent",
                    "phase": "complete",
                    "input": messages,
                    "output": output,
                    "input_tokens": self._token_count(processor, prompt),
                    "output_tokens": generated,
                    "max_tokens": self._max_tokens(),
                    "context_window": model_context_window(self.settings),
                    "selected_id": selected_id,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
            return selected_id
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_selection = {"parsed": False, "error": type(exc).__name__}
            self.emit_model_event(
                {
                    "id": operation_id,
                    "source": "agent",
                    "phase": "error",
                    "input": messages,
                    "output": output,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": self.last_error,
                }
            )
            return None
