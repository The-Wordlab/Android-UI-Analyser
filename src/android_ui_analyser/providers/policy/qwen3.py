"""Optional local Qwen3 selector over already-guarded AUA policy candidates.

This provider exists because a capacity comparison needs the challenger to run in the same lane
as the incumbent, not only in an offline harness. It is text-only, receives the identical
privacy-screened candidate projection as FunctionGemma, and never sees screenshots, hierarchy
dumps, typed values, or trusted call arguments. It loads only an existing absolute MLX model
directory and never downloads weights.

Two details differ from FunctionGemma, and both are transport rather than contract:

* **Activation turn.** FunctionGemma carries its activation on a ``developer`` role. Qwen3's chat
  template accepts only system/user/assistant/tool and raises on anything else, so the identical
  activation text is rendered on a ``system`` turn — the same single-field change the Qwen training
  corpus uses. Rendering must match training exactly or the adapter is being asked a question in a
  shape it never saw.
* **Verdict envelope.** Qwen emits ``<tool_call>{...}</tool_call>``. The contract is unchanged:
  exactly one ``select_candidate`` call carrying one integer, and nothing after it.

Everything that decides safety — candidate compilation, guarding, post-inference revalidation —
stays outside this class, exactly as it does for the other providers.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import re
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...policy import POLICY_HANDOFF_ID, PolicyContext, guard_candidates, policy_messages
from ...policy import policy_tools as _policy_tools
from ..base import Availability, PolicyProvider
from ..registry import register_policy

# One tool call, optionally preceded by a thinking block, and nothing after it.
_TOOL_CALL = re.compile(
    r"\s*(?:<think>.*?</think>\s*)?<tool_call>\s*(\{.*?\})\s*(?:</tool_call>)?\s*",
    re.DOTALL,
)
_ACTIVATION_ROLE = "system"
#: The pip extra that installs this selector's inference runtime (mlx-lm). Shared with
#: the FunctionGemma provider — one extra covers both mlx-lm selectors.
INSTALL_EXTRA = "functiongemma"


def _absolute_model_directory(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("models.qwen3.model_path must be an absolute local directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("models.qwen3.model_path must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("models.qwen3.model_path must be a directory")
    if not (resolved / "config.json").is_file():
        raise ValueError("Qwen3 model directory is missing config.json")
    if not any(resolved.glob("*.safetensors")):
        raise ValueError("Qwen3 model directory contains no safetensors weights")
    return resolved


def _absolute_adapter_directory(value: Any) -> Path | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError("models.qwen3.adapter_path must be an absolute local directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("models.qwen3.adapter_path must be absolute")
    resolved = path.resolve(strict=True)
    if not (resolved / "adapters.safetensors").is_file():
        raise ValueError("Qwen3 adapter directory is missing adapters.safetensors")
    return resolved


def parse_tool_call(output: str) -> int:
    """Parse one exact ``select_candidate`` verdict, rejecting prose or trailing output."""

    match = _TOOL_CALL.fullmatch(output)
    if match is None:
        raise ValueError("Qwen3 output is not one exact tool call")
    payload = json.loads(match.group(1))
    if not isinstance(payload, Mapping) or payload.get("name") != "select_candidate":
        raise ValueError("Qwen3 output calls an unexpected function")
    arguments = payload.get("arguments")
    value = arguments.get("candidate_id") if isinstance(arguments, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("candidate_id is not an integer")
    return value


@register_policy("qwen3")
class Qwen3PolicySelector(PolicyProvider):
    """Lazy, resident MLX Qwen3 selector over already-guarded opaque candidates."""

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
        self._tokenizer: Any | None = None
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
        value = self.settings.get("max_tokens", 48)
        if not isinstance(value, int) or isinstance(value, bool) or not 16 <= value <= 512:
            raise ValueError("models.qwen3.max_tokens must be an integer from 16 to 512")
        return value

    def _runtime_availability(self) -> Availability:
        if self._model_loader is not None and self._generator is not None:
            return Availability(True, "injected runtime")
        if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            return Availability(False, "Qwen3 MLX policy requires Apple silicon")
        if importlib.util.find_spec("mlx_lm") is None:
            # Name the extra, not the wheel: "install mlx-lm" into a `uv tool`
            # environment by hand is the exact move the next upgrade silently undoes.
            return Availability(
                False,
                f"optional dependency missing; install android-ui-analyser[{INSTALL_EXTRA}]",
            )
        return Availability(True, "local MLX-LM runtime available")

    def is_available(self) -> Availability:
        try:
            runtime = self._runtime_availability()
            if not runtime.ok:
                return runtime
            _absolute_model_directory(self.settings.get("model_path"))
            _absolute_adapter_directory(self.settings.get("adapter_path"))
            self._max_tokens()
        except Exception as exc:
            self.last_error = str(exc)
            return Availability(False, str(exc))
        self.last_error = None
        return Availability(True, "local Qwen3 selector is ready")

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
            "supported_candidate_counts": [2, 3, 4],
            "supports_handoff": True,
            "activation_role": _ACTIVATION_ROLE,
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
                "adapter_path": self.settings.get("adapter_path"),
            },
            "last_error": self.last_error,
            "last_selection": dict(self.last_selection),
        }

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer
            model_path = _absolute_model_directory(self.settings.get("model_path"))
            adapter_path = _absolute_adapter_directory(self.settings.get("adapter_path"))
            if self._model_loader is None or self._generator is None:
                from mlx_lm import generate, load

                self._model_loader = self._model_loader or load
                self._generator = self._generator or generate
            loader = self._model_loader
            if loader is None:
                raise RuntimeError("Qwen3 model loader is unavailable")
            kwargs: dict[str, Any] = {}
            if adapter_path is not None:
                kwargs["adapter_path"] = str(adapter_path)
            loaded = loader(str(model_path), **kwargs)
            if not isinstance(loaded, tuple) or len(loaded) < 2:
                raise RuntimeError("Qwen3 model loader returned an invalid result")
            self._model, self._tokenizer = loaded[:2]
        return self._model, self._tokenizer

    def _render(self, context: PolicyContext, tokenizer: Any) -> str:
        messages = policy_messages(context)
        # Qwen3's template has no developer role; the training corpus carries the identical
        # activation text on a system turn, and inference must match it exactly.
        messages = [
            {**message, "role": _ACTIVATION_ROLE} if message["role"] == "developer" else message
            for message in messages
        ]
        return tokenizer.apply_chat_template(
            messages,
            tools=_policy_tools(allow_handoff=context.allow_handoff),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    def select(self, context: PolicyContext) -> int | None:
        try:
            guarded = guard_candidates(
                context,
                max_candidates=max(1, min(4, len(context.candidates))),
            )
            if guarded != context.candidates:
                raise ValueError("provider received candidates that did not pass the policy guard")
            if not self.supports_candidate_count(len(guarded)):
                raise ValueError("unsupported candidate cardinality")
            model, tokenizer = self._ensure_loaded()
            prompt = self._render(context, tokenizer)
            generator = self._generator
            if generator is None:
                raise RuntimeError("Qwen3 generator is unavailable")
            with self._generation_lock:
                result = generator(
                    model, tokenizer, prompt, max_tokens=self._max_tokens(), verbose=False
                )
            output = str(getattr(result, "text", result))
            selected_id = parse_tool_call(output)
            offered = {candidate.candidate_id for candidate in guarded}
            if selected_id != POLICY_HANDOFF_ID and selected_id not in offered:
                raise ValueError("Qwen3 selected an ID outside the guarded candidates")
            self.last_selection = {"parsed": True, "selected_id": selected_id}
            self.last_error = None
            return selected_id
        except Exception as exc:
            # Shape only: no model text, candidate label, or resource id is retained.
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_selection = {"parsed": False, "error": type(exc).__name__}
            return None
