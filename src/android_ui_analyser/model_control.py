"""Runtime control and local-only telemetry for AUA's resident policy models.

The dashboard is a different process from the warm daemon that owns the model instances.  A
small cache-backed control document therefore carries operator switches across that boundary,
while an append-only bounded event stream lets the dashboard observe inference without routing
model prompts through the device or the agent journal.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text

MODEL_NAMES = ("functiongemma", "gemma4")
_STATE_SCHEMA = 1
_EVENT_LIMIT = 300
_TEXT_LIMIT = 100_000


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Keep localhost diagnostics useful without allowing an unbounded cache record."""

    if depth > 8:
        return "<depth-limited>"
    if isinstance(value, str):
        return value[:_TEXT_LIMIT]
    if isinstance(value, Mapping):
        return {
            str(key)[:200]: _bounded(item, depth=depth + 1)
            for key, item in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(item, depth=depth + 1) for item in value[:200]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_TEXT_LIMIT]


def model_context_window(settings: Mapping[str, Any]) -> int | None:
    """Read a local model's declared context window without loading its weights."""

    raw = settings.get("model_path")
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        return None
    path = Path(raw).expanduser() / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    value = config.get("max_position_embeddings")
    if not isinstance(value, int):
        text_config = config.get("text_config")
        value = (
            text_config.get("max_position_embeddings") if isinstance(text_config, dict) else None
        )
    return value if isinstance(value, int) and value > 0 else None


class ModelControlStore:
    """Shared runtime switches plus a bounded, process-safe-enough local event stream."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.root = Path(config.cache.dir).expanduser() / "model-control"
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self._lock = threading.RLock()

    def _configured_enabled(self) -> bool:
        return bool(getattr(getattr(self.config, "policy", None), "enabled", False))

    def read_state(self) -> dict[str, Any]:
        default = {
            "schema_version": _STATE_SCHEMA,
            "intercept_enabled": self._configured_enabled(),
            "providers": dict.fromkeys(MODEL_NAMES, True),
            "source": "config",
            "updated_ms": None,
        }
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return default
        if not isinstance(value, dict) or value.get("schema_version") != _STATE_SCHEMA:
            return default
        providers = value.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        return {
            "schema_version": _STATE_SCHEMA,
            "intercept_enabled": bool(value.get("intercept_enabled", False)),
            "providers": {name: bool(providers.get(name, True)) for name in MODEL_NAMES},
            "source": "dashboard",
            "updated_ms": value.get("updated_ms"),
        }

    def update(
        self,
        *,
        intercept_enabled: bool | None = None,
        provider: str | None = None,
        provider_enabled: bool | None = None,
    ) -> dict[str, Any]:
        if provider is not None and provider not in MODEL_NAMES:
            raise ValueError(f"unknown local model {provider!r}")
        if provider is not None and provider_enabled is None:
            raise ValueError("provider_enabled is required with provider")
        with self._lock:
            state = self.read_state()
            if intercept_enabled is not None:
                state["intercept_enabled"] = bool(intercept_enabled)
            if provider is not None:
                state["providers"][provider] = bool(provider_enabled)
            state["schema_version"] = _STATE_SCHEMA
            state["updated_ms"] = int(time.time() * 1000)
            state.pop("source", None)
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self.state_path,
                json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n",
            )
        return self.read_state()

    def intercept_enabled(self) -> bool:
        return bool(self.read_state()["intercept_enabled"])

    def intercept_override(self) -> bool | None:
        state = self.read_state()
        if state.get("source") != "dashboard":
            return None
        return bool(state["intercept_enabled"])

    def provider_enabled(self, name: str) -> bool:
        return bool(self.read_state()["providers"].get(name, True))

    def record(self, event: Mapping[str, Any]) -> dict[str, Any]:
        value = {
            "id": str(event.get("id") or uuid.uuid4().hex),
            "timestamp_ms": int(event.get("timestamp_ms") or time.time() * 1000),
            **_bounded(dict(event)),
        }
        line = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._trim_events()
        return value

    def _trim_events(self) -> None:
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= _EVENT_LIMIT:
            return
        atomic_write_text(self.events_path, "\n".join(lines[-_EVENT_LIMIT:]) + "\n")

    def events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in lines[-max(1, min(int(limit), _EVENT_LIMIT)) :]:
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                out.append(value)
        return out

    def clear_events(self) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.events_path, "")
