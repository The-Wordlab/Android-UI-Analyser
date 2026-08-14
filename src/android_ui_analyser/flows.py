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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .atomic import atomic_create_text, atomic_write_text
from .errors import UsageError
from .memory import REDACT_TOKENS, RouteStep, _safe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import MemoryCfg

FLOW_SCHEMA_VERSION = 1

_PARAM_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ARRIVAL_FIELDS = frozenset({"text", "rid", "id", "desc", "net", "log"})

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
    "network_offline": "network-offline",
    "network_restore": "network-restore",
    "network_profile": "network-profile",
    "network_profile_restore": "network-profile-restore",
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
    "network-profile": "name",
}
# Accepted by `scroll_to: {direction: ...}` — the same vocabulary `_swipe_path` takes, so the
# flow surface and the CLI's `--direction` mean exactly one thing between them.
_SCROLL_DIRECTIONS = frozenset({"up", "down", "left", "right"})
_A11Y_SCROLL_DIRECTIONS = frozenset({"forward", "fwd", "down", "backward", "back", "up"})
_DEV_PROFILES = frozenset({"ac", "default"})
_NETWORK_PROFILES = frozenset({"wifi-only", "cellular-only", "slow", "lossy"})
# Keep this vocabulary aligned with Engine.key.  Importing Engine here would create a cycle:
# Engine deliberately imports the flow parser only at the flow execution boundary.
_KEY_NAMES = frozenset(
    {
        "home",
        "back",
        "left",
        "right",
        "up",
        "down",
        "center",
        "menu",
        "search",
        "enter",
        "delete",
        "del",
        "recent",
        "recents",
        "volume_up",
        "volume_down",
        "volume_mute",
        "camera",
        "power",
    }
)
_BARE_KINDS = frozenset(
    {
        "wait-stable",
        "launch-app",
        "stop-app",
        "hide-keyboard",
        "paste",
        "proxy-start",
        "proxy-stop",
        "network-offline",
        "network-restore",
        "network-profile-restore",
    }
)


ArrivalStatus = Literal["mapped", "predicate_verified", "unverified"]


class Flow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: int = FLOW_SCHEMA_VERSION
    name: str
    app: str | None = None  # primary package: origin for package-relative steps / goto
    context_id: str | None = None  # recorded deterministic memory/feature context
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    arrival: str | None = None
    arrival_screen: str | None = None  # mapped-screen proof, recognized during replay
    arrival_status: ArrivalStatus | None = None
    params: dict[str, str] = Field(default_factory=dict)  # "" = required, else default
    steps: list[RouteStep]

    @model_validator(mode="after")
    def _arrival_contract(self) -> Flow:
        if self.arrival_status == "mapped" and not self.arrival_screen:
            raise ValueError("arrival_status `mapped` requires `arrival_screen`")
        if self.arrival_status == "predicate_verified" and not self.arrival:
            raise ValueError("arrival_status `predicate_verified` requires `arrival`")
        if self.arrival_status == "unverified" and self.arrival_screen:
            raise ValueError("arrival_status `unverified` cannot claim `arrival_screen`")
        if self.arrival_screen and self.arrival_status != "mapped":
            raise ValueError("arrival_screen requires arrival_status `mapped`")
        return self


class SelectorResilience(BaseModel):
    """Value-free disclosure of how well one selector survives a later replay."""

    model_config = ConfigDict(extra="forbid")
    step: str | None = None
    selector: Literal["rid", "desc", "text", "id", "composite", "none"]
    strength: Literal["strong", "medium", "weak", "frame_only", "unknown", "missing"]
    localization_risk: bool
    cross_frame: bool
    index_sensitive: bool = False
    reason: str


def describe_selector_resilience(
    *,
    rid: str | None = None,
    desc: str | None = None,
    text: str | None = None,
    element_id: int | None = None,
    index: int | None = None,
    step: str | None = None,
    legacy_composite: bool = False,
) -> SelectorResilience:
    """Classify a selector without echoing its potentially private value.

    Resource ids normally survive both frame churn and localization. Content descriptions are
    semantic but may be translated; visible text is the most translation/copy-sensitive replay
    selector. AUA's integer element id is intentionally valid only for the current observation,
    so it must never be presented as a saved-flow selector.
    """
    positional = index is not None
    selector: Literal["rid", "desc", "text", "id", "composite", "none"]
    strength: Literal["strong", "medium", "weak", "frame_only", "unknown", "missing"]
    if legacy_composite and any((rid, desc, text)):
        selector = "composite"
        strength = "unknown"
        localization_risk = bool(desc or text)
        cross_frame = False
        reason = (
            "legacy selector may fall back across resource id, description, and visible text; "
            "no single replay tier or cross-frame guarantee was recorded"
        )
    elif rid:
        selector = "rid"
        strength = "strong"
        localization_risk = False
        cross_frame = True
        reason = "resource id is stable across observations and independent of translated copy"
    elif desc:
        selector = "desc"
        strength = "medium"
        localization_risk = True
        cross_frame = True
        reason = "content description is semantic but may change with localization or copy"
    elif text:
        selector = "text"
        strength = "weak"
        localization_risk = True
        cross_frame = True
        reason = "visible text is replayable but sensitive to localization and copy changes"
    elif element_id is not None:
        selector = "id"
        strength = "frame_only"
        localization_risk = False
        cross_frame = False
        reason = "integer element id belongs only to the observation that produced it"
    else:
        selector = "none"
        strength = "missing"
        localization_risk = False
        cross_frame = False
        reason = "no semantic selector was captured"
    if positional:
        reason += "; positional index also depends on match ordering"
    return SelectorResilience(
        step=step,
        selector=selector,
        strength=strength,
        localization_risk=localization_risk,
        cross_frame=cross_frame,
        index_sensitive=positional,
        reason=reason,
    )


def recorded_selector_resilience(
    steps: Sequence[RouteStep], where: str = ""
) -> list[SelectorResilience]:
    """Describe every recorded element selector, including nested flow blocks."""
    out: list[SelectorResilience] = []
    selector_kinds = {"tap", "long-press", "input", "clear", "a11y-scroll"}
    for index, route_step in enumerate(steps, start=1):
        at = f"{where}step {index}" if where else f"step {index}"
        if route_step.kind in selector_kinds:
            selected = route_step.by
            out.append(
                describe_selector_resilience(
                    rid=(route_step.resource_id if selected in {None, "id", "rid"} else None),
                    desc=(route_step.content_desc if selected in {None, "desc"} else None),
                    text=(route_step.label if selected in {None, "text"} else None),
                    index=route_step.index,
                    step=at,
                    legacy_composite=selected is None,
                )
            )
        if route_step.substeps:
            out.extend(recorded_selector_resilience(route_step.substeps, f"{at} > "))
    return out


# --------------------------------------------------------------------------- parsing


def _step_error(index: int, msg: str, hint: str | None = None) -> UsageError:
    return UsageError(f"flow step {index + 1}: {msg}", hint=hint)


def _strict_int(value: Any, *, field: str, index: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _step_error(index, f"{field} must be an integer, got {value!r}")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise _step_error(index, f"{field} must be {qualifier}, got {value!r}")
    return value


def _strict_bool(value: Any, *, field: str, index: int) -> bool:
    if not isinstance(value, bool):
        raise _step_error(index, f"{field} must be true or false, got {value!r}")
    return value


def _string(value: Any, *, field: str, index: int, optional: bool = True) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise _step_error(index, f"{field} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise _step_error(index, f"{field} must not be empty")
    return value


def _key_string(value: Any, *, field: str, index: int) -> str:
    """Return a key name in the same accepted vocabulary as :meth:`Engine.key`.

    YAML decodes an unquoted numeric keycode as an integer.  Accept that one lossless scalar
    conversion while retaining the parser's refusal to coerce booleans, floats, or containers.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    parsed = _string(value, field=field, index=index, optional=False)
    assert parsed is not None  # optional=False
    candidate = parsed.strip()
    if not (
        candidate.lower() in _KEY_NAMES
        or candidate.upper().startswith("KEYCODE_")
        or candidate.isdigit()
    ):
        raise _step_error(
            index,
            f"{field} is not a supported Android key: {value!r}",
            hint=(
                "Use one of " + ", ".join(sorted(_KEY_NAMES)) + ", KEYCODE_*, or a numeric keycode."
            ),
        )
    return candidate


def _tap_point(value: str, *, field: str, index: int) -> str:
    """Validate and canonicalize the coordinate grammar consumed by Engine._parse_point."""
    parts = value.replace(" ", "").split(",")
    if len(parts) == 2:
        try:
            x, y = (int(round(float(part))) for part in parts)
        except (OverflowError, ValueError):
            pass
        else:
            if x >= 0 and y >= 0:
                return f"{x},{y}"
    raise _step_error(
        index,
        f"{field} must be two non-negative coordinates written as `x,y`, got {value!r}",
    )


def _normalize_arg(kind: str, value: str, *, field: str, index: int) -> str:
    """Fail at YAML load time when Engine would reject a known finite-vocabulary argument."""
    # A flow template is not executable yet.  Preserve placeholder-bearing arguments until
    # ``resolve_params`` supplies their values, then validate the fully materialized graph with
    # ``validate_resolved_steps`` before any caller can execute it.
    if _PARAM_RE.search(value):
        return value
    if kind == "key":
        return _key_string(value, field=field, index=index)
    if kind == "tap-point":
        return _tap_point(value, field=field, index=index)
    if kind in {"swipe", "scroll"}:
        normalized = value.strip().lower()
        if normalized not in _SCROLL_DIRECTIONS:
            raise _step_error(
                index,
                f"{field} must be one of {', '.join(sorted(_SCROLL_DIRECTIONS))}, got {value!r}",
            )
        return normalized
    if kind == "a11y-scroll":
        normalized = value.strip().lower()
        if normalized not in _A11Y_SCROLL_DIRECTIONS:
            raise _step_error(
                index,
                f"{field} must be one of "
                f"{', '.join(sorted(_A11Y_SCROLL_DIRECTIONS))}, got {value!r}",
            )
        return normalized
    if kind == "dev-profile":
        normalized = value.strip().lower()
        if normalized not in _DEV_PROFILES:
            raise _step_error(
                index,
                f"{field} must be `ac` or `default`, got {value!r}",
            )
        return normalized
    if kind == "network-profile":
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in _NETWORK_PROFILES:
            raise _step_error(
                index,
                f"{field} must be one of {', '.join(sorted(_NETWORK_PROFILES))}, got {value!r}",
            )
        return normalized
    return value


def _top_string(data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise UsageError(f"flow `{field}:` must be a string, got {type(value).__name__}")
    if not value.strip():
        raise UsageError(f"flow `{field}:` must not be empty")
    return value


def _arrival_status(data: dict[str, Any]) -> ArrivalStatus | None:
    value = _top_string(data, "arrival_status")
    if value is None:
        return None
    if value == "mapped":
        return "mapped"
    if value == "predicate_verified":
        return "predicate_verified"
    if value == "unverified":
        return "unverified"
    raise UsageError(
        "flow `arrival_status:` must be `mapped`, `predicate_verified`, or `unverified`"
    )


def validate_arrival_predicate(predicate: str) -> None:
    """Validate the action-bound arrival grammar without importing :mod:`engine`.

    This is the save-time half of the same deliberately small comma-separated grammar used by
    runtime awaits. A reusable flow must contain positive evidence that something arrived;
    absence-only conditions such as ``!text:Loading`` are useful waits but cannot prove arrival.
    """
    raw = (predicate or "").strip()
    if not raw:
        raise UsageError("flow arrival needs a predicate")

    chunks: list[str] = []
    current: list[str] = []
    escaped = False
    for char in raw:
        if escaped:
            if char in {",", "\\"}:
                current.append(char)
            else:
                current.extend(("\\", char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ",":
            chunks.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        raise UsageError("flow arrival predicate ends with an incomplete escape")
    chunks.append("".join(current))

    terms = 0
    positive = False
    for chunk in chunks:
        piece = chunk.strip()
        if not piece:
            continue
        negated = piece.startswith("!")
        body = piece[1:].strip() if negated else piece
        prefix, separator, value = body.partition(":")
        if not separator or not value.strip():
            raise UsageError(f"flow arrival term {piece!r} needs a <field>:<value> form")
        field = prefix.strip().lower()
        if field not in _ARRIVAL_FIELDS:
            raise UsageError(f"flow arrival term {piece!r} names an unknown field {field!r}")
        terms += 1
        positive = positive or not negated
    if not terms:
        raise UsageError("flow arrival needs at least one term")
    if not positive:
        raise UsageError(
            "flow arrival needs at least one positive arrival term",
            hint="Add text:, rid:, desc:, net:, or log: evidence that proves arrival.",
        )


def _parse_step(item: Any, index: int) -> RouteStep:
    if isinstance(item, dict) and len(item) == 1:
        ((key, value),) = item.items()
        if key in ("repeat", "retry"):
            if not isinstance(value, dict):
                raise _step_error(index, f"{key} needs a mapping with `steps:`")
            composite = dict(value)
            raw_sub = composite.pop("steps", None)
            commands = composite.pop("commands", None)
            if raw_sub is not None and commands is not None:
                raise _step_error(index, f"{key} accepts `steps:` or `commands:`, not both")
            raw_sub = raw_sub if raw_sub is not None else commands
            if not isinstance(raw_sub, list) or not raw_sub:
                raise _step_error(index, f"{key} needs a non-empty `steps:` list")
            substeps = [_parse_step(sub, index * 100 + j) for j, sub in enumerate(raw_sub)]
            if key == "repeat":
                raw_times = composite.pop("times", None)
                raw_count = composite.pop("count", None)
                if raw_times is not None and raw_count is not None:
                    raise _step_error(index, "repeat accepts `times:` or `count:`, not both")
                count = raw_times if raw_times is not None else raw_count
                times = (
                    1
                    if count is None
                    else _strict_int(count, field="repeat count", index=index, minimum=1)
                )
                if composite:
                    raise _step_error(
                        index,
                        "unknown keys for repeat: " + ", ".join(sorted(map(str, composite))),
                    )
                return RouteStep(kind="repeat", repeat=times, substeps=substeps)
            retry_values = [
                (field, composite.pop(field))
                for field in ("max_retries", "maxRetries", "times")
                if field in composite
            ]
            if len(retry_values) > 1:
                raise _step_error(
                    index, "retry accepts only one of `max_retries:`, `maxRetries:`, or `times:`"
                )
            max_retries = (
                3
                if not retry_values
                else _strict_int(
                    retry_values[0][1], field="retry max_retries", index=index, minimum=1
                )
            )
            if composite:
                raise _step_error(
                    index,
                    "unknown keys for retry: " + ", ".join(sorted(map(str, composite))),
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
        raise _step_error(index, 'expected a single-key mapping like `tap: "Send"`')
    ((key, value),) = item.items()
    if not isinstance(key, str):
        raise _step_error(index, f"step kind must be a string, got {type(key).__name__}")
    kind = _KINDS.get(key)
    if kind is None:
        raise _step_error(
            index,
            f"unknown step kind {key!r}",
            hint="known kinds: " + ", ".join(sorted(_KINDS)),
        )

    if value is None:
        value = {}
    if kind == "key" and isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if isinstance(value, str):
        if kind in _ELEMENT_KINDS:
            selector = _string(value, field=f"{key} selector", index=index, optional=False)
            assert selector is not None  # optional=False
            return RouteStep(kind=kind, label=selector)
        if kind in _ARG_ALIAS:
            arg = _string(value, field=f"{key} {_ARG_ALIAS[kind]}", index=index, optional=False)
            assert arg is not None  # optional=False
            return RouteStep(
                kind=kind,
                arg=_normalize_arg(
                    kind,
                    arg,
                    field=f"{key} {_ARG_ALIAS[kind]}",
                    index=index,
                ),
            )
        if kind == "wait-stable":
            return RouteStep(kind=kind)
        raise _step_error(index, "input needs a mapping: `input: {id: ..., text: ...}`")
    if not isinstance(value, dict):
        raise _step_error(
            index, f"step value must be a string or mapping, got {type(value).__name__}"
        )

    v = dict(value)
    kw: dict[str, Any] = {"kind": kind}
    if kind in _ELEMENT_KINDS:
        explicit_by = v.pop("by", None)
        if explicit_by not in (None, "id", "desc", "text"):
            raise _step_error(index, f"{key} `by:` must be id, desc, or text")
        kw["resource_id"] = _string(v.pop("id", None), field=f"{key} id", index=index)
        kw["content_desc"] = _string(v.pop("desc", None), field=f"{key} desc", index=index)
        raw_text = v.pop("text", None)
        raw_label = v.pop("label", None)
        if raw_text is not None and raw_label is not None:
            raise _step_error(index, f"{key} accepts `text:` or `label:`, not both")
        kw["label"] = _string(
            raw_text if raw_text is not None else raw_label,
            field=f"{key} text",
            index=index,
        )
        kw["by"] = explicit_by
        if not (kw["resource_id"] or kw["content_desc"] or kw["label"]):
            raise _step_error(index, f"{key} needs an `id:`, `desc:`, or `text:` selector")
        if explicit_by == "id" and not kw["resource_id"]:
            raise _step_error(index, f"{key} `by: id` needs an `id:` selector")
        if explicit_by == "desc" and not kw["content_desc"]:
            raise _step_error(index, f"{key} `by: desc` needs a `desc:` selector")
        if explicit_by == "text" and not kw["label"]:
            raise _step_error(index, f"{key} `by: text` needs a `text:` selector")
        if (nth := v.pop("index", None)) is not None:
            # Coercion is refused rather than applied: silently reading 1.5 as "the second
            # match" is the class of quiet guess `index:` was added to remove.
            if isinstance(nth, bool) or not isinstance(nth, int):
                raise _step_error(
                    index, f"{key} `index:` must be a whole number (0-based), got {nth!r}"
                )
            if nth < 0:
                raise _step_error(index, f"{key} `index:` must not be negative, got {nth!r}")
            kw["index"] = nth
    elif kind == "input":
        explicit_by = v.pop("by", None)
        if explicit_by not in (None, "id", "desc", "text"):
            raise _step_error(index, "input `by:` must be id, desc, or text")
        kw["resource_id"] = _string(v.pop("id", None), field="input id", index=index)
        kw["content_desc"] = _string(v.pop("desc", None), field="input desc", index=index)
        kw["label"] = _string(v.pop("label", None), field="input label", index=index)
        raw_input = v.pop("text", None)
        if raw_input is not None and not isinstance(raw_input, str):
            raise _step_error(index, f"input text must be a string, got {type(raw_input).__name__}")
        kw["text"] = raw_input
        kw["submit"] = _strict_bool(v.pop("submit", False), field="input submit", index=index)
        if kw["text"] is None:
            raise _step_error(index, "input needs `text:` (a literal or ${PARAM})")
        kw["by"] = explicit_by
        if not (kw["resource_id"] or kw["content_desc"] or kw["label"]):
            raise _step_error(index, "input needs an `id:`, `desc:`, or `label:` field selector")
        if explicit_by == "id" and not kw["resource_id"]:
            raise _step_error(index, "input `by: id` needs an `id:` selector")
        if explicit_by == "desc" and not kw["content_desc"]:
            raise _step_error(index, "input `by: desc` needs a `desc:` selector")
        if explicit_by == "text" and not kw["label"]:
            raise _step_error(index, "input `by: text` needs a `label:` selector")
    elif kind in (
        "wait-stable",
        "hide-keyboard",
        "paste",
        "proxy-start",
        "proxy-stop",
        "network-offline",
        "network-restore",
        "network-profile-restore",
    ):
        pass
    elif kind == "a11y-scroll":
        explicit_by = v.pop("by", None)
        if explicit_by not in (None, "id", "desc", "text"):
            raise _step_error(index, "a11y_scroll `by:` must be id, desc, or text")
        raw_id = v.pop("id", None)
        raw_rid = v.pop("rid", None)
        if raw_id is not None and raw_rid is not None:
            raise _step_error(index, "a11y_scroll accepts `id:` or `rid:`, not both")
        kw["resource_id"] = _string(
            raw_id if raw_id is not None else raw_rid,
            field="a11y_scroll id",
            index=index,
        )
        kw["content_desc"] = _string(v.pop("desc", None), field="a11y_scroll desc", index=index)
        raw_text = v.pop("text", None)
        raw_label = v.pop("label", None)
        if raw_text is not None and raw_label is not None:
            raise _step_error(index, "a11y_scroll accepts `text:` or `label:`, not both")
        kw["label"] = _string(
            raw_text if raw_text is not None else raw_label,
            field="a11y_scroll text",
            index=index,
        )
        kw["by"] = explicit_by
        if explicit_by == "id" and not kw["resource_id"]:
            raise _step_error(index, "a11y_scroll `by: id` needs an `id:` selector")
        if explicit_by == "desc" and not kw["content_desc"]:
            raise _step_error(index, "a11y_scroll `by: desc` needs a `desc:` selector")
        if explicit_by == "text" and not kw["label"]:
            raise _step_error(index, "a11y_scroll `by: text` needs a `text:` selector")
        direction = _string(
            v.pop("direction", "forward"),
            field="a11y_scroll direction",
            index=index,
            optional=False,
        )
        assert direction is not None  # optional=False
        kw["arg"] = _normalize_arg(
            kind,
            direction,
            field="a11y_scroll direction",
            index=index,
        )
        if not (kw["resource_id"] or kw["content_desc"] or kw["label"]):
            raise _step_error(
                index, "a11y_scroll needs an `id:`/`rid:`, `desc:`, or `text:` selector"
            )
    elif kind in ("launch-app", "stop-app"):
        # Optional arg: a bare `stop_app`/`launch_app` targets the flow's own app.
        raw_alias = v.pop(_ARG_ALIAS[kind], None)
        raw_arg = v.pop("arg", None)
        if raw_alias is not None and raw_arg is not None:
            raise _step_error(index, f"{key} accepts `{_ARG_ALIAS[kind]}:` or `arg:`, not both")
        kw["arg"] = _string(
            raw_alias if raw_alias is not None else raw_arg,
            field=f"{key} {_ARG_ALIAS[kind]}",
            index=index,
        )
        # `launch_app: {activity: ...}` pins the entry Activity. Needed on builds that
        # declare more than one MAIN/LAUNCHER component (a dev flavour shipping a
        # developer-tools launcher alongside the product one), where letting the system
        # resolve the launcher is a coin toss and the following wait then times out on a
        # screen the flow never meant to be on.
        if kind == "launch-app":
            kw["activity"] = _string(
                v.pop("activity", None), field="launch_app activity", index=index
            )
    else:
        alias = _ARG_ALIAS[kind]
        raw_alias = v.pop(alias, None)
        raw_arg = v.pop("arg", None)
        if raw_alias is not None and raw_arg is not None:
            raise _step_error(index, f"{key} accepts `{alias}:` or `arg:`, not both")
        raw_value = raw_alias if raw_alias is not None else raw_arg
        if kind == "key" and isinstance(raw_value, int) and not isinstance(raw_value, bool):
            raw_value = str(raw_value)
        parsed_arg = _string(raw_value, field=f"{key} {alias}", index=index)
        kw["arg"] = (
            _normalize_arg(kind, parsed_arg, field=f"{key} {alias}", index=index)
            if parsed_arg is not None
            else None
        )
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
            rid = _string(v.pop("id", None), field=f"{key} id", index=index)
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
            if not isinstance(way, str) or (
                not _PARAM_RE.search(way) and way.lower() not in _SCROLL_DIRECTIONS
            ):
                raise _step_error(
                    index,
                    f"{key} `direction:` must be one of "
                    f"{', '.join(sorted(_SCROLL_DIRECTIONS))}, got {way!r}",
                    hint="`up` looks further down the list (the default); `down` looks back up.",
                )
            kw["direction"] = way if _PARAM_RE.search(way) else way.lower()
    kw["package"] = _string(v.pop("package", None), field=f"{key} package", index=index)
    timeout = v.pop("timeout_ms", None)
    if timeout is not None:
        kw["timeout_ms"] = _strict_int(timeout, field=f"{key} timeout_ms", index=index, minimum=0)
    if v:
        raise _step_error(index, f"unknown keys for {key}: {', '.join(sorted(map(str, v)))}")
    try:
        return RouteStep(**kw)
    except ValidationError as exc:
        raise _step_error(index, f"invalid {key}: {exc.errors()[0]['msg']}") from exc


def parse_flow_yaml(text: str, *, name: str | None = None) -> Flow:
    """Parse the agent-facing YAML into a :class:`Flow` (UsageError on any bad step)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise UsageError(f"flow YAML does not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("flow YAML must be a mapping with a `steps:` list")
    if any(not isinstance(key, str) for key in data):
        raise UsageError("every top-level flow key must be a string")
    known_top = {
        "schema_version",
        "name",
        "app",
        "context_id",
        "description",
        "aliases",
        "arrival",
        "arrival_screen",
        "arrival_status",
        "params",
        "steps",
    }
    if unknown := sorted(key for key in data if key not in known_top):
        raise UsageError("unknown top-level flow keys: " + ", ".join(map(str, unknown)))
    schema_version = data.get("schema_version", FLOW_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise UsageError("flow `schema_version:` must be an integer")
    if schema_version != FLOW_SCHEMA_VERSION:
        raise UsageError(
            f"unsupported flow schema_version {schema_version}; expected {FLOW_SCHEMA_VERSION}"
        )
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise UsageError("flow needs a non-empty `steps:` list")
    steps = [_parse_step(item, i) for i, item in enumerate(raw_steps)]
    params = data.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise UsageError("`params:` must be a mapping of NAME: default (empty = required)")
    normalized_params: dict[str, str] = {}
    for param_name, default in params.items():
        if not isinstance(param_name, str) or not param_name.strip():
            raise UsageError("every `params:` name must be a non-empty string")
        if default is not None and not isinstance(default, str):
            raise UsageError(f"flow param `{param_name}` default must be a string or null")
        normalized_params[param_name] = "" if default is None else default
    aliases = data.get("aliases", [])
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list):
        raise UsageError("`aliases:` must be a list of goal phrases")
    if any(not isinstance(value, str) or not value.strip() for value in aliases):
        raise UsageError("every `aliases:` value must be a non-empty string")
    declared_name = _top_string(data, "name")
    if declared_name is None:
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise UsageError("flow name must be a non-empty string")
        declared_name = name or "flow"
    try:
        return Flow(
            schema_version=schema_version,
            name=declared_name,
            app=_top_string(data, "app"),
            context_id=_top_string(data, "context_id"),
            description=_top_string(data, "description"),
            aliases=aliases,
            arrival=_top_string(data, "arrival"),
            arrival_screen=_top_string(data, "arrival_screen"),
            arrival_status=_arrival_status(data),
            params=normalized_params,
            steps=steps,
        )
    except ValidationError as exc:
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise UsageError(f"invalid flow: {detail}") from exc


# --------------------------------------------------------------------------- rendering


def _render_step(s: RouteStep) -> dict[str, Any] | str:
    if s.kind == "repeat":
        if not s.substeps or not s.repeat or s.repeat < 1:
            raise UsageError("repeat step needs a positive count and non-empty substeps")
        return {
            "repeat": {
                "times": s.repeat,
                "steps": [_render_step(substep) for substep in s.substeps],
            }
        }
    if s.kind == "retry":
        if not s.substeps or not s.max_retries or s.max_retries < 1:
            raise UsageError("retry step needs positive max_retries and non-empty substeps")
        return {
            "retry": {
                "max_retries": s.max_retries,
                "steps": [_render_step(substep) for substep in s.substeps],
            }
        }
    try:
        key = _KEYS[s.kind]
    except KeyError as exc:
        raise UsageError(f"flow step kind {s.kind!r} cannot be rendered or replayed") from exc
    extras: dict[str, Any] = {}
    if s.package:
        extras["package"] = s.package
    if s.timeout_ms is not None:
        extras["timeout_ms"] = s.timeout_ms
    if s.kind == "launch-app" and s.activity:
        extras["activity"] = s.activity
    if s.kind == "scroll-to" and s.direction:
        # Must round-trip: `check_saveable` re-parses its own rendering, so a direction that
        # rendered away would be silently dropped by the very check that proves a flow loads.
        extras["direction"] = s.direction
    if s.kind in _ELEMENT_KINDS:
        body: dict[str, Any] = {}
        if s.resource_id:
            body["id"] = s.resource_id
        if s.content_desc:
            body["desc"] = s.content_desc
        if s.label:
            body["text"] = s.label
        if s.by:
            body["by"] = s.by
        if s.index is not None:
            body["index"] = s.index
        body.update(extras)
        if list(body) == ["text"] and s.by is None:
            return {key: s.label}
        return {key: body}
    if s.kind == "input":
        body = {}
        if s.resource_id:
            body["id"] = s.resource_id
        if s.content_desc:
            body["desc"] = s.content_desc
        if s.label:
            body["label"] = s.label
        if s.by:
            body["by"] = s.by
        body["text"] = s.text or ""
        if s.submit:
            body["submit"] = True
        body.update(extras)
        return {key: body}
    if s.kind == "a11y-scroll":
        body = {"direction": s.arg or "forward"}
        if s.resource_id:
            body["rid"] = s.resource_id
        if s.content_desc:
            body["desc"] = s.content_desc
        if s.label:
            body["text"] = s.label
        if s.by:
            body["by"] = s.by
        body.update(extras)
        return {key: body}
    if s.kind in (
        "wait-stable",
        "proxy-start",
        "proxy-stop",
        "network-offline",
        "network-restore",
        "network-profile-restore",
        "hide-keyboard",
        "paste",
    ):
        return {key: extras} if extras else key
    if extras:
        return {key: {_ARG_ALIAS[s.kind]: s.arg, **extras}}
    return {key: s.arg}


def render_flow_yaml(flow: Flow) -> str:
    doc: dict[str, Any] = {"schema_version": flow.schema_version, "name": flow.name}
    if flow.app:
        doc["app"] = flow.app
    if flow.context_id:
        doc["context_id"] = flow.context_id
    if flow.description:
        doc["description"] = flow.description
    if flow.aliases:
        doc["aliases"] = flow.aliases
    if flow.arrival:
        doc["arrival"] = flow.arrival
    if flow.arrival_screen:
        doc["arrival_screen"] = flow.arrival_screen
    if flow.arrival_status:
        doc["arrival_status"] = flow.arrival_status
    if flow.params:
        doc["params"] = dict(flow.params)
    doc["steps"] = [_render_step(s) for s in flow.steps]
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------- params


def validate_resolved_steps(
    steps: list[RouteStep], *, allow_unresolved_params: bool = False
) -> list[RouteStep]:
    """Validate and canonicalize a fully materialized step graph without side effects.

    Parse-time validation covers literal YAML. Parameterized finite-vocabulary arguments cannot
    be judged until substitution, so every caller that resolves a flow receives this recursive
    gate before execution. The public helper also gives nested-flow loaders the same pure boundary
    when they already hold materialized :class:`RouteStep` objects. Save validation alone may set
    ``allow_unresolved_params`` after proving every remaining placeholder is a declared required
    parameter with an intentionally empty default.
    """

    def validate(step: RouteStep, index: int) -> RouteStep:
        update: dict[str, Any] = {}
        if step.kind in {
            "key",
            "tap-point",
            "swipe",
            "scroll",
            "dev-profile",
            "network-profile",
        }:
            if not step.arg:
                raise _step_error(index, f"{step.kind} needs a non-empty argument")
            if _PARAM_RE.search(step.arg):
                if not allow_unresolved_params:
                    raise _step_error(
                        index, f"{step.kind} argument contains an unresolved parameter"
                    )
            else:
                update["arg"] = _normalize_arg(
                    step.kind,
                    step.arg,
                    field=f"{_KEYS[step.kind]} {_ARG_ALIAS[step.kind]}",
                    index=index,
                )
        elif step.kind == "a11y-scroll":
            direction = step.arg or "forward"
            if _PARAM_RE.search(direction):
                if not allow_unresolved_params:
                    raise _step_error(
                        index, "a11y_scroll direction contains an unresolved parameter"
                    )
            else:
                update["arg"] = _normalize_arg(
                    step.kind,
                    direction,
                    field="a11y_scroll direction",
                    index=index,
                )

        if step.kind == "scroll-to" and step.direction is not None:
            direction = step.direction
            if _PARAM_RE.search(direction):
                if not allow_unresolved_params:
                    raise _step_error(index, "scroll_to direction contains an unresolved parameter")
            else:
                normalized = direction.strip().lower()
                if normalized not in _SCROLL_DIRECTIONS:
                    raise _step_error(
                        index,
                        "scroll_to direction must be one of "
                        f"{', '.join(sorted(_SCROLL_DIRECTIONS))}, got {direction!r}",
                    )
                update["direction"] = normalized

        if step.substeps:
            update["substeps"] = [
                validate(substep, index * 100 + child_index)
                for child_index, substep in enumerate(step.substeps)
            ]
        return step.model_copy(update=update) if update else step

    return [validate(step, index) for index, step in enumerate(steps)]


def _substitute_step_params(
    steps: list[RouteStep], values: dict[str, str]
) -> tuple[list[RouteStep], set[str]]:
    """Materialize known values recursively, retaining and reporting unknown placeholders."""
    unresolved: set[str] = set()

    def sub(text: str | None) -> str | None:
        if not text:
            return text

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in values or values[name] == "":
                unresolved.add(name)
                return match.group(0)
            return values[name]

        return _PARAM_RE.sub(repl, text)

    def fix(step: RouteStep) -> RouteStep:
        return step.model_copy(
            update={
                "label": sub(step.label),
                "content_desc": sub(step.content_desc),
                "resource_id": sub(step.resource_id),
                "text": sub(step.text),
                "arg": sub(step.arg),
                "package": sub(step.package),
                "activity": sub(step.activity),
                "direction": sub(step.direction),
                "substeps": [fix(substep) for substep in step.substeps],
            }
        )

    return [fix(step) for step in steps], unresolved


def resolve_params(flow: Flow, given: dict[str, str]) -> list[RouteStep]:
    """Substitute ``${NAME}`` in label/text/arg; UsageError names anything unresolved."""
    values = {k: given.get(k, v) for k, v in flow.params.items()}
    values.update(given)
    missing = sorted(k for k, v in values.items() if v == "" and k in flow.params)
    steps, unresolved = _substitute_step_params(flow.steps, values)
    problems = sorted(set(missing) | unresolved)
    if problems:
        raise UsageError(
            "missing flow param(s): " + ", ".join(problems),
            hint="pass --param NAME=value (declare defaults under `params:`)",
        )
    return validate_resolved_steps(steps)


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
    params: dict[str, str] = {}
    n = 0

    def fix(s: RouteStep) -> RouteStep:
        nonlocal n
        update: dict[str, Any] = {}
        if s.kind == "input":
            n += 1
            params[f"PARAM_{n}"] = ""
            update["text"] = f"${{PARAM_{n}}}"
        elif s.label in REDACT_TOKENS or s.content_desc in REDACT_TOKENS:
            n += 1
            params[f"PARAM_{n}"] = ""
            if s.content_desc in REDACT_TOKENS:
                update["content_desc"] = f"${{PARAM_{n}}}"
            else:
                update["label"] = f"${{PARAM_{n}}}"
        if s.substeps:
            update["substeps"] = [fix(substep) for substep in s.substeps]
        return s.model_copy(update=update) if update else s

    return [fix(step) for step in recent], params


def recorded_step_blockers(steps: list[RouteStep], where: str = "") -> list[str]:
    """Explain why recorded actions cannot be replayed losslessly.

    This validator is intentionally capture-specific. Agent-authored YAML may use the
    corresponding flow steps because it supplies their complete arguments. The action
    journal currently records only the reduced :class:`RouteStep` shape, so gestures whose
    container, percentage, repetition, coordinates, matching mode, or action name was lost
    must never be advertised as an exact reusable capture.
    """
    out: list[str] = []
    lossy = {
        "long-press": "long-press capture omits the requested hold duration",
        "double-tap": "double-tap has no flow replay step",
        "a11y-action": "accessibility action name cannot be replayed by a flow",
        "swipe": "swipe capture omits coordinates/container/percentage",
        "scroll": "scroll capture omits container/pages/end-condition/percentage",
        "scroll-to": "scroll-to capture omits match/case/direction/limit/percentage",
        "paste": "paste capture omits the clipboard value it depends on",
        "open-link": "open-link capture omits package-pinning and chooser policy",
    }
    replayable = {
        "tap",
        "tap-point",
        "input",
        "clear",
        "key",
        "hide-keyboard",
        "a11y-scroll",
        "dev-profile",
        "flags-apply",
        "mock-replay",
        "network-offline",
        "network-restore",
        "network-profile",
        "network-profile-restore",
    }
    selector_kinds = {"tap", "long-press", "input", "clear", "a11y-scroll"}
    for index, step in enumerate(steps, start=1):
        at = f"{where}step {index}" if where else f"step {index}"
        if reason := lossy.get(step.kind):
            out.append(f"{at} ({step.kind}): {reason}")
        elif step.kind not in replayable and step.kind not in {"repeat", "retry"}:
            out.append(f"{at} ({step.kind}): recorded action kind is not exactly replayable")
        elif step.kind in selector_kinds and not (
            step.resource_id or step.content_desc or step.label
        ):
            out.append(
                f"{at} ({step.kind}) has no unique stable id, safe description, "
                "or safe visible text selector"
            )
        elif step.kind == "tap-point" and not re.fullmatch(r"\d+,\d+", step.arg or ""):
            out.append(f"{at} (tap-point): exact coordinates were not captured")
        elif step.kind == "key" and not step.arg:
            out.append(f"{at} (key): key name was not captured")
        elif step.kind == "a11y-scroll" and (step.arg or "").lower() not in {
            "forward",
            "fwd",
            "down",
            "backward",
            "back",
            "up",
        }:
            out.append(f"{at} (a11y-scroll): direction was not captured exactly")
        if step.kind in {"repeat", "retry"}:
            if not step.substeps:
                out.append(f"{at} ({step.kind}): composite capture has no substeps")
            else:
                out.extend(recorded_step_blockers(step.substeps, f"{at} > "))
    return out


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
        for value in (s.label, s.content_desc, s.resource_id):
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

    def referenced_params(steps: list[RouteStep]) -> set[str]:
        found: set[str] = set()
        for s in steps:
            for value in (
                s.label,
                s.content_desc,
                s.resource_id,
                s.text,
                s.arg,
                s.package,
                s.activity,
                s.direction,
            ):
                if value:
                    found.update(_PARAM_RE.findall(value))
            if s.substeps:
                found.update(referenced_params(s.substeps))
        return found

    referenced = referenced_params(reparsed.steps)
    if undeclared := sorted(referenced - declared):
        raise UsageError(
            "refusing to save a flow with unbound parameter(s): " + ", ".join(undeclared),
            hint="nothing can supply them, so `flow run` would fail before touching the device",
        )

    if reparsed.arrival:
        validate_arrival_predicate(reparsed.arrival)

    default_values = {name: value for name, value in reparsed.params.items() if value != ""}
    default_steps = reparsed.steps
    # Defaults may themselves refer to another declared default. Materialize to a fixed point;
    # one pass per declared value is sufficient for an acyclic chain. A cycle remains visible as
    # a non-required placeholder and is refused below rather than being saved for a late failure.
    for _ in range(len(default_values) + 1):
        materialized, _unresolved = _substitute_step_params(default_steps, default_values)
        if materialized == default_steps:
            break
        default_steps = materialized
    required = {name for name, value in reparsed.params.items() if value == ""}
    if unresolved_defaults := sorted(referenced_params(default_steps) - required):
        raise UsageError(
            "refusing to save a flow with unresolved default parameter(s): "
            + ", ".join(unresolved_defaults),
            hint="use literal defaults, or leave a directly referenced parameter empty and required",
        )
    validate_resolved_steps(
        default_steps,
        allow_unresolved_params=bool(referenced_params(default_steps)),
    )

    warnings = _selector_warnings(default_steps)
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

    def list(
        self,
        *,
        active_package: str | None = None,
        active_context_id: str | None = None,
    ) -> list[dict[str, Any]]:
        d = self.flows_dir()
        if not d.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for p in sorted(d.glob("*.yaml")):
            try:
                flow = parse_flow_yaml(p.read_text(encoding="utf-8"), name=p.stem)
                compatible: bool | None = None
                if active_package is not None and active_context_id is not None:
                    compatible = flow.app in (None, active_package) and flow.context_id in (
                        None,
                        active_context_id,
                    )
                out.append(
                    {
                        "name": flow.name,
                        # The declared display name may intentionally differ from the file
                        # name. Execution is keyed by the storage name, so discovery must
                        # expose both instead of recommending a command that cannot load.
                        "storage_name": p.stem,
                        "app": flow.app,
                        "context_id": flow.context_id,
                        "steps": len(flow.steps),
                        "params": sorted(flow.params),
                        "description": flow.description,
                        "aliases": flow.aliases,
                        "arrival": flow.arrival,
                        "arrival_screen": flow.arrival_screen,
                        "arrival_status": flow.arrival_status or "unverified",
                        "context_compatible": compatible,
                        "path": str(p),
                    }
                )
            except Exception as exc:
                out.append(
                    {
                        "name": p.stem,
                        "storage_name": p.stem,
                        "error": str(exc),
                        "path": str(p),
                    }
                )
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
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_flow_yaml(flow)
        if force:
            atomic_write_text(path, rendered)
            return path
        try:
            atomic_create_text(path, rendered)
        except FileExistsError as exc:
            raise UsageError(
                f"flow '{flow.name}' already exists", hint="pass --force to overwrite"
            ) from exc
        return path

    def delete(self, name: str) -> bool:
        path = self.path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
