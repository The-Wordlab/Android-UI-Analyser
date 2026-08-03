"""Named flows — Maestro-style journeys the CLI replays in one call (PRD §6b).

A flow is a YAML file of :class:`~.memory.RouteStep`-shaped steps, either **authored by
an agent directly** (it may never have walked the UI) or **materialized** from the
session's recent actions with ``aua flow save``. ``aua flow run <name>`` executes the
whole journey — taps, waits, asserts, cross-app auth legs — through the same step
executor ``goto`` uses, handing back a resumable step index on divergence. The point is
fewer agent iterations: the boring path to the screen under test becomes one call.

Design notes
------------
- **Flat namespace.** Flows live at ``<memory.dir>/flows/<name>.yaml`` with the primary
  app recorded *inside* (``app:``) — journeys span packages by design (Google auth runs
  in Chrome), so scoping files by package would make cross-app flows homeless.
- **Privacy.** ``flow save`` never sees typed values (recording redacts them); inputs
  and redacted labels are materialized as ``${PARAM_n}`` placeholders for the agent to
  fill. Authored flows MAY carry literal text — that is their purpose (e.g. a test
  account label); they are local files under the user's memory dir.
- **Params.** ``${NAME}`` placeholders substitute from declared ``params:`` defaults
  overridden by ``--param NAME=value``; an empty default means required.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .errors import UsageError
from .memory import REDACT_TOKENS, RouteStep, _safe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import MemoryCfg

FLOW_SCHEMA_VERSION = 1

_PARAM_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# YAML step key (snake_case) → RouteStep.kind (kebab-case).
_KINDS = {
    "tap": "tap",
    "long_press": "long-press",
    "input": "input",
    "clear": "clear",
    "key": "key",
    "swipe": "swipe",
    "scroll_to": "scroll-to",
    "wait_for": "wait-for",
    "wait_stable": "wait-stable",
    "assert_visible": "assert-visible",
    "assert_not_visible": "assert-not-visible",
    "hide_keyboard": "hide-keyboard",
    "paste": "paste",
    "launch_app": "launch-app",
    "stop_app": "stop-app",
    "open_link": "open-link",
    "goto": "goto",
    "run_flow": "flow",  # alias; canonical render key is `flow` (listed last → wins _KEYS)
    "flow": "flow",
    "dev_profile": "dev-profile",
    "a11y_scroll": "a11y-scroll",
    "flags_apply": "flags-apply",
    "proxy_start": "proxy-start",
    "proxy_stop": "proxy-stop",
    "mock_replay": "mock-replay",
}
_KEYS = {kind: key for key, kind in _KINDS.items()}
_ELEMENT_KINDS = ("tap", "long-press", "clear")
# For arg-carrying kinds, the natural mapping-form key name.
_ARG_ALIAS = {
    "key": "name",
    "swipe": "direction",
    "scroll-to": "text",
    "wait-for": "text",
    "assert-visible": "text",
    "assert-not-visible": "text",
    "launch-app": "package",
    "stop-app": "package",
    "open-link": "uri",
    "goto": "screen",
    "flow": "name",
    "dev-profile": "name",
    "flags-apply": "path",
    "mock-replay": "name",
}
_BARE_KINDS = frozenset(
    {
        "wait-stable",
        "launch-app",
        "stop-app",
        "hide-keyboard",
        "paste",
        "proxy-start",
        "proxy-stop",
    }
)


class Flow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: int = FLOW_SCHEMA_VERSION
    name: str
    app: str | None = None  # primary package: origin for package-relative steps / goto
    description: str | None = None
    params: dict[str, str] = Field(default_factory=dict)  # "" = required, else default
    steps: list[RouteStep]


# --------------------------------------------------------------------------- parsing


def _step_error(index: int, msg: str, hint: str | None = None) -> UsageError:
    return UsageError(f"flow step {index + 1}: {msg}", hint=hint)


def _parse_step(item: Any, index: int) -> RouteStep:
    if isinstance(item, dict) and len(item) == 1:
        ((key, value),) = item.items()
        if key in ("repeat", "retry"):
            if not isinstance(value, dict):
                raise _step_error(index, f"{key} needs a mapping with `steps:`")
            raw_sub = value.get("steps") or value.get("commands") or []
            if not isinstance(raw_sub, list) or not raw_sub:
                raise _step_error(index, f"{key} needs a non-empty `steps:` list")
            substeps = [_parse_step(sub, index * 100 + j) for j, sub in enumerate(raw_sub)]
            if key == "repeat":
                times = int(value.get("times") or value.get("count") or 1)
                return RouteStep(kind="repeat", repeat=times, substeps=substeps)
            max_retries = int(
                value.get("max_retries") or value.get("maxRetries") or value.get("times") or 3
            )
            return RouteStep(kind="retry", max_retries=max_retries, substeps=substeps)
    if isinstance(item, str):
        # Bare-string steps that need no argument (like Maestro's `- stopApp`).
        if _KINDS.get(item) in _BARE_KINDS:
            return RouteStep(kind=_KINDS[item])
        raise _step_error(
            index,
            f"a bare string step must be one of {', '.join(sorted(_BARE_KINDS))}, got {item!r}",
        )
    if not isinstance(item, dict) or len(item) != 1:
        raise _step_error(index, "expected a single-key mapping like `tap: \"Send\"`")
    ((key, value),) = item.items()
    kind = _KINDS.get(str(key))
    if kind is None:
        raise _step_error(
            index,
            f"unknown step kind {key!r}",
            hint="known kinds: " + ", ".join(sorted(_KINDS)),
        )

    if value is None:
        value = {}
    if isinstance(value, str):
        if kind in _ELEMENT_KINDS:
            return RouteStep(kind=kind, label=value)
        if kind in _ARG_ALIAS:
            return RouteStep(kind=kind, arg=value)
        if kind == "wait-stable":
            return RouteStep(kind=kind)
        raise _step_error(index, "input needs a mapping: `input: {id: ..., text: ...}`")
    if not isinstance(value, dict):
        raise _step_error(index, f"step value must be a string or mapping, got {type(value).__name__}")

    v = dict(value)
    kw: dict[str, Any] = {"kind": kind}
    if kind in _ELEMENT_KINDS:
        kw["resource_id"] = v.pop("id", None)
        kw["label"] = v.pop("text", None) or v.pop("desc", None) or v.pop("label", None)
        if not (kw["resource_id"] or kw["label"]):
            raise _step_error(index, f"{key} needs an `id:` or `text:` selector")
    elif kind == "input":
        kw["resource_id"] = v.pop("id", None)
        kw["label"] = v.pop("label", None)
        kw["text"] = v.pop("text", None)
        kw["submit"] = bool(v.pop("submit", False))
        if kw["text"] is None:
            raise _step_error(index, "input needs `text:` (a literal or ${PARAM})")
        if not (kw["resource_id"] or kw["label"]):
            raise _step_error(index, "input needs an `id:` or `label:` field selector")
    elif kind in ("wait-stable", "hide-keyboard", "paste", "proxy-start", "proxy-stop"):
        pass
    elif kind == "a11y-scroll":
        kw["resource_id"] = v.pop("id", None) or v.pop("rid", None)
        kw["label"] = v.pop("text", None) or v.pop("label", None)
        kw["arg"] = str(v.pop("direction", None) or "forward")
        if not (kw["resource_id"] or kw["label"]):
            raise _step_error(index, "a11y_scroll needs an `id:`/`rid:` or `text:` selector")
    elif kind in ("launch-app", "stop-app"):
        # Optional arg: a bare `stop_app`/`launch_app` targets the flow's own app.
        kw["arg"] = v.pop(_ARG_ALIAS[kind], None) or v.pop("arg", None)
        # `launch_app: {activity: ...}` pins the entry Activity. Needed on builds that
        # declare more than one MAIN/LAUNCHER component (a dev flavour shipping a
        # developer-tools launcher alongside the product one), where letting the system
        # resolve the launcher is a coin toss and the following wait then times out on a
        # screen the flow never meant to be on.
        if kind == "launch-app":
            kw["activity"] = v.pop("activity", None)
    else:
        alias = _ARG_ALIAS[kind]
        kw["arg"] = v.pop(alias, None) or v.pop("arg", None)
        # scroll_to/wait_for/assert_visible/assert_not_visible may target a resource-id:
        # `{id: containerX}`. assert_not_visible needs it as much as its positive twin —
        # "this id is gone" is how you check a selected tab (which drops its rid) or an
        # entry point that must not be offered on a given screen.
        if kw["arg"] is None and kind in (
            "scroll-to",
            "wait-for",
            "assert-visible",
            "assert-not-visible",
        ):
            rid = v.pop("id", None)
            if rid is not None:
                kw["arg"] = rid
                kw["by"] = "id"
        if not kw["arg"]:
            raise _step_error(index, f"{key} needs `{alias}:` (or `id:` for a resource-id)")
    kw["package"] = v.pop("package", None)
    timeout = v.pop("timeout_ms", None)
    if timeout is not None:
        kw["timeout_ms"] = int(timeout)
    if v:
        raise _step_error(index, f"unknown keys for {key}: {', '.join(sorted(map(str, v)))}")
    return RouteStep(**kw)


def parse_flow_yaml(text: str, *, name: str | None = None) -> Flow:
    """Parse the agent-facing YAML into a :class:`Flow` (UsageError on any bad step)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise UsageError(f"flow YAML does not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("flow YAML must be a mapping with a `steps:` list")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise UsageError("flow needs a non-empty `steps:` list")
    steps = [_parse_step(item, i) for i, item in enumerate(raw_steps)]
    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise UsageError("`params:` must be a mapping of NAME: default (empty = required)")
    return Flow(
        name=str(data.get("name") or name or "flow"),
        app=data.get("app"),
        description=data.get("description"),
        params={str(k): "" if v is None else str(v) for k, v in params.items()},
        steps=steps,
    )


# --------------------------------------------------------------------------- rendering


def _render_step(s: RouteStep) -> dict[str, Any] | str:
    key = _KEYS[s.kind]
    extras: dict[str, Any] = {}
    if s.package:
        extras["package"] = s.package
    if s.timeout_ms is not None:
        extras["timeout_ms"] = s.timeout_ms
    if s.kind in _ELEMENT_KINDS:
        body: dict[str, Any] = {}
        if s.resource_id:
            body["id"] = s.resource_id
        if s.label:
            body["text"] = s.label
        body.update(extras)
        if list(body) == ["text"]:
            return {key: s.label}
        return {key: body}
    if s.kind == "input":
        body = {}
        if s.resource_id:
            body["id"] = s.resource_id
        if s.label:
            body["label"] = s.label
        body["text"] = s.text or ""
        if s.submit:
            body["submit"] = True
        body.update(extras)
        return {key: body}
    if s.kind == "a11y-scroll":
        body = {"direction": s.arg or "forward"}
        if s.resource_id:
            body["rid"] = s.resource_id
        if s.label:
            body["text"] = s.label
        body.update(extras)
        return {key: body}
    if s.kind in ("wait-stable", "proxy-start", "proxy-stop", "hide-keyboard", "paste"):
        return {key: extras} if extras else key
    if extras:
        return {key: {_ARG_ALIAS[s.kind]: s.arg, **extras}}
    return {key: s.arg}


def render_flow_yaml(flow: Flow) -> str:
    doc: dict[str, Any] = {"schema_version": flow.schema_version, "name": flow.name}
    if flow.app:
        doc["app"] = flow.app
    if flow.description:
        doc["description"] = flow.description
    if flow.params:
        doc["params"] = dict(flow.params)
    doc["steps"] = [_render_step(s) for s in flow.steps]
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------- params


def resolve_params(flow: Flow, given: dict[str, str]) -> list[RouteStep]:
    """Substitute ``${NAME}`` in label/text/arg; UsageError names anything unresolved."""
    values = {k: given.get(k, v) for k, v in flow.params.items()}
    values.update(given)
    missing = sorted(k for k, v in values.items() if v == "" and k in flow.params)
    unresolved: set[str] = set()

    def sub(text: str | None) -> str | None:
        if not text:
            return text

        def repl(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in values or values[name] == "":
                unresolved.add(name)
                return m.group(0)
            return values[name]

        return _PARAM_RE.sub(repl, text)

    steps = [
        s.model_copy(update={"label": sub(s.label), "text": sub(s.text), "arg": sub(s.arg)})
        for s in flow.steps
    ]
    problems = sorted(set(missing) | unresolved)
    if problems:
        raise UsageError(
            "missing flow param(s): " + ", ".join(problems),
            hint="pass --param NAME=value (declare defaults under `params:`)",
        )
    return steps


def steps_from_recent(recent: list[RouteStep]) -> tuple[list[RouteStep], dict[str, str]]:
    """Materialize recorded steps for ``flow save``: redacted values → ``${PARAM_n}``.

    Typed values were never recorded, so every input becomes a required parameter; a
    redacted tap label (PII) becomes one too — the agent fills them in the saved file.
    """
    out: list[RouteStep] = []
    params: dict[str, str] = {}
    n = 0
    for s in recent:
        if s.kind == "input":
            n += 1
            params[f"PARAM_{n}"] = ""
            s = s.model_copy(update={"text": f"${{PARAM_{n}}}"})
        elif s.label in REDACT_TOKENS:
            n += 1
            params[f"PARAM_{n}"] = ""
            s = s.model_copy(update={"label": f"${{PARAM_{n}}}"})
        out.append(s)
    return out, params


# --------------------------------------------------------------------------- store


class FlowStore:
    """Read/write named flows under ``<memory.dir>/flows/`` (flat namespace)."""

    def __init__(self, cfg: MemoryCfg) -> None:
        self.cfg = cfg

    def flows_dir(self) -> Path:
        return Path(self.cfg.dir).expanduser() / "flows"

    def path(self, name: str) -> Path:
        return self.flows_dir() / f"{_safe(name)}.yaml"

    def list(self) -> list[dict[str, Any]]:
        d = self.flows_dir()
        if not d.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for p in sorted(d.glob("*.yaml")):
            try:
                flow = parse_flow_yaml(p.read_text(encoding="utf-8"), name=p.stem)
                out.append(
                    {
                        "name": flow.name,
                        "app": flow.app,
                        "steps": len(flow.steps),
                        "params": sorted(flow.params),
                        "description": flow.description,
                        "path": str(p),
                    }
                )
            except UsageError as exc:
                out.append({"name": p.stem, "error": str(exc), "path": str(p)})
        return out

    def load(self, name: str) -> Flow:
        path = self.path(name)
        if not path.is_file():
            known = ", ".join(sorted(p.stem for p in self.flows_dir().glob("*.yaml"))) or "(none)"
            raise UsageError(f"no flow named '{name}'", hint=f"known flows: {known}")
        return parse_flow_yaml(path.read_text(encoding="utf-8"), name=name)

    def save(self, flow: Flow, *, force: bool = False) -> Path:
        path = self.path(flow.name)
        if path.exists() and not force:
            raise UsageError(f"flow '{flow.name}' already exists", hint="pass --force to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_flow_yaml(flow), encoding="utf-8")
        return path

    def delete(self, name: str) -> bool:
        path = self.path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
