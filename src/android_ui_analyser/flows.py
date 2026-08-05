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
    "tap_point": "tap-point",
    "long_press": "long-press",
    "input": "input",
    "clear": "clear",
    "key": "key",
    "swipe": "swipe",
    "scroll": "scroll",
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
    "tap-point": "point",
    "key": "name",
    "swipe": "direction",
    "scroll": "direction",
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
# Accepted by `scroll_to: {direction: ...}` — the same vocabulary `_swipe_path` takes, so the
# flow surface and the CLI's `--direction` mean exactly one thing between them.
_SCROLL_DIRECTIONS = frozenset({"up", "down", "left", "right"})
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
        if (nth := v.pop("index", None)) is not None:
            # Coercion is refused rather than applied: silently reading 1.5 as "the second
            # match" is the class of quiet guess `index:` was added to remove.
            if isinstance(nth, bool) or not isinstance(nth, int):
                if not (isinstance(nth, str) and nth.isdigit()):
                    raise _step_error(
                        index, f"{key} `index:` must be a whole number (0-based), got {nth!r}"
                    )
                nth = int(nth)
            if nth < 0:
                raise _step_error(index, f"{key} `index:` must not be negative, got {nth!r}")
            kw["index"] = nth
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
        # `scroll_to` searches ONE way (default: swipe up, i.e. look further down the list), and
        # the step could not say which. A tool grid that opens already scrolled past its target
        # therefore had the search go away from it, and the flow failed live validation as though
        # the card were absent — a search that went the wrong way looks exactly like a missing
        # element, so it invites "the card is gone" instead of "I searched away from it". The
        # workaround was an explicit `swipe: down` first, which only works by luck of distance.
        if kind == "scroll-to" and (way := v.pop("direction", None)) is not None:
            if str(way).lower() not in _SCROLL_DIRECTIONS:
                raise _step_error(
                    index,
                    f"{key} `direction:` must be one of "
                    f"{', '.join(sorted(_SCROLL_DIRECTIONS))}, got {way!r}",
                    hint="`up` looks further down the list (the default); `down` looks back up.",
                )
            kw["direction"] = str(way).lower()
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
    if s.kind == "scroll-to" and s.direction:
        # Must round-trip: `check_saveable` re-parses its own rendering, so a direction that
        # rendered away would be silently dropped by the very check that proves a flow loads.
        extras["direction"] = s.direction
    if s.kind in _ELEMENT_KINDS:
        body: dict[str, Any] = {}
        if s.resource_id:
            body["id"] = s.resource_id
        if s.label:
            body["text"] = s.label
        if s.index is not None:
            body["index"] = s.index
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


# Step kinds whose `arg` is a host filesystem path rather than a name or a label.
# `mock-replay` and `dev-profile` take *names*, so they must not be touched. A nested `flow:`
# takes either — see :func:`looks_like_path` — and is resolved at execution time rather than
# here, because the candidate list includes directories `anchor_paths` cannot know about.
_PATH_KINDS = frozenset({"flags-apply"})


def looks_like_path(ref: str) -> bool:
    """Is this nested-``flow:`` reference a path rather than a name?

    Nested flows resolved by name from AUA's own memory directory only, so promoting shared
    preconditions into `flows/common/` and referencing them from `flows/derived/*` was
    impossible: a promoted flow that referenced a sibling broke for anyone whose memory
    directory did not happen to contain a flow of that name. Nine shared routes therefore had
    to be *inlined* into ~35 derived flows, so the same steps exist in many copies and a fix to
    one does not propagate. `grep` keeps them in step, which is a convention, not a guarantee.

    The test has to be conservative in one specific direction: a name that is mistaken for a
    path merely fails to resolve and says so, while a *path* mistaken for a name is looked up
    in the memory directory under a sanitised spelling — where it could match some unrelated
    flow and silently run the wrong journey. So this asks for positive evidence of a path
    (a separator, a YAML suffix, a `~`, or an explicit `./`), and a bare word stays a name.
    """
    text = str(ref or "").strip()
    if not text:
        return False
    if text.startswith("~") or Path(text).is_absolute():
        return True
    if "/" in text or "\\" in text:
        return True
    return text.lower().endswith((".yaml", ".yml"))


def nested_flow_candidates(
    ref: str, referring_dir: Path | None, memory_flows_dir: Path | None
) -> list[Path]:
    """Where a path-looking nested ``flow:`` reference may live, in precedence order.

    ``flows/derived/x.yaml`` saying ``flow: common/auth.yaml`` means "next to me" first — that
    is the reading that makes a flow directory portable, which is the whole point of keeping
    flows in a repository. A reference relative to the *collection* root is the second reading,
    so the nearest enclosing directory named ``flows`` is tried next (that is how
    ``derived/a.yaml`` reaches ``common/auth.yaml`` without spelling `../`). The memory
    directory comes last, so nothing that resolves inside the repository can be shadowed by
    whatever happens to be installed on one machine.
    """
    text = str(ref).strip()
    path = Path(text).expanduser()
    if path.is_absolute():
        return [path]
    out: list[Path] = []
    if referring_dir is not None:
        base = Path(referring_dir).expanduser().resolve()
        out.append(base / path)
        # Walk up to the nearest `flows` collection root, including `base` itself.
        for parent in [base, *base.parents]:
            if parent.name == "flows":
                out.append(parent / path)
                break
    if memory_flows_dir is not None:
        out.append(Path(memory_flows_dir).expanduser() / path)
    # Preserve order, drop repeats (a flow directly inside `flows/` yields the same candidate).
    seen: set[Path] = set()
    unique: list[Path] = []
    for cand in out:
        if cand not in seen:
            seen.add(cand)
            unique.append(cand)
    return unique


def anchor_paths(steps: list[RouteStep], base_dir: Path) -> list[RouteStep]:
    """Resolve a step's relative host path against *base_dir* — the flow file's directory.

    A flow that says ``flags_apply: flags/guest.yaml`` means "next to me". It cannot mean
    "relative to whatever directory the caller happened to be in", and it certainly cannot
    mean "relative to the daemon's cwd", which is what it got: the reporting lane had to
    rewrite the reference to an absolute path to make the flow run at all.

    Anchoring here also makes a flow directory portable — it can be checked in, moved, and
    run from anywhere, which is the whole point of keeping flows in a repository.

    Call this *after* param substitution, so `${DIR}/flags.yaml` anchors the value the
    caller supplied rather than the placeholder.
    """

    def fix(step: RouteStep) -> RouteStep:
        update: dict[str, Any] = {}
        if step.kind in _PATH_KINDS and step.arg:
            path = Path(step.arg).expanduser()
            if not path.is_absolute():
                update["arg"] = str((base_dir / path).resolve())
        if step.substeps:
            update["substeps"] = [fix(sub) for sub in step.substeps]
        return step.model_copy(update=update) if update else step

    return [fix(s) for s in steps]


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


# ------------------------------------------------------------------- save validation

# A selector built from any of these will match on the visit that recorded it and never
# again: a clock reading, a rendered file size, or a backend-generated identifier. They
# come from list rows that put volatile detail in the content-desc — a document picker
# publishing "report.pdf, 1.4 MB, 09:42" is one selector that is really three facts, two
# of which change.
_VOLATILE_SELECTOR = (
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), "a wall-clock time"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "a date"),
    (re.compile(r"\b\d+(?:[.,]\d+)?\s?[KMGT]?B\b"), "a file size"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "a uuid"),
    (re.compile(r"\b(?=[0-9a-z]*\d)[0-9a-f]{16,}\b"), "a backend-looking id"),
)


def _selector_warnings(steps: list[RouteStep], where: str = "") -> list[str]:
    out: list[str] = []
    for i, s in enumerate(steps, start=1):
        at = f"{where}step {i}" if where else f"step {i}"
        for value in (s.label, s.resource_id):
            if not value:
                continue
            for pattern, what in _VOLATILE_SELECTOR:
                if pattern.search(value):
                    out.append(f"{at}: selector contains {what} and will not match on a later run")
                    break
        if s.substeps:
            out.extend(_selector_warnings(s.substeps, f"{at} > "))
    return out


def check_saveable(flow: Flow) -> list[str]:
    """Reject a flow that cannot execute; return warnings for one that merely might not.

    ``flow save`` used to write whatever it had recorded, so a capture could produce a file
    that read plausibly and died on first use — which is worse than no file, because the
    capture step reports success either way.
    """
    try:
        reparsed = parse_flow_yaml(render_flow_yaml(flow), name=flow.name)
    except UsageError as exc:
        raise UsageError(
            f"refusing to save a flow that cannot be loaded back: {exc}",
            hint="the recorded step is missing a selector `flow run` needs — drive it again",
        ) from exc

    declared = set(reparsed.params)
    referenced: set[str] = set()

    def scan(steps: list[RouteStep]) -> None:
        for s in steps:
            for value in (s.label, s.text, s.arg):
                if value:
                    referenced.update(_PARAM_RE.findall(value))
            if s.substeps:
                scan(s.substeps)

    scan(reparsed.steps)
    if undeclared := sorted(referenced - declared):
        raise UsageError(
            "refusing to save a flow with unbound parameter(s): " + ", ".join(undeclared),
            hint="nothing can supply them, so `flow run` would fail before touching the device",
        )

    warnings = _selector_warnings(reparsed.steps)
    if empty := sorted(k for k in declared if reparsed.params[k] == ""):
        warnings.append(
            "declared with no value, so `flow run` fails until each is filled in or passed "
            "with --param: " + ", ".join(empty)
        )
    return warnings


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
        check_saveable(flow)  # never write an artefact that cannot run
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
