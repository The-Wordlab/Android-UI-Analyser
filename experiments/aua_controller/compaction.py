"""Model-facing frame compaction: keep what a controller or judge needs, drop the rest.

Raw evidence is never changed. This shapes only the text a model reads, and is applied
after the privacy projection. A real application screen carries ~20 fields per element
and ~20 metadata keys; a controller acts on ids, labels and a few state flags, so the
rest is paid for on every request without informing a decision. An unchanged screen
collapses to a fingerprint stub with interactive handles only, so re-analysis of the
same frame does not resend it.
"""

from __future__ import annotations

import copy
from typing import Any

ELEMENT_FIELDS = (
    "id", "text", "desc", "content_desc", "resource_id", "rid", "bounds",
    "clickable", "editable", "checked", "selected", "scrollable", "focused", "window",
)
STATE_FLAGS = ("clickable", "editable", "scrollable")
META_FIELDS = ("fingerprint", "stale_risk", "changed", "known_screen", "arrival_state",
               # What the app asked for and has not been answered. A loading screen and a
               # finished one are the same hierarchy; this is the only field that separates
               # them, so dropping it here would hide the whole signal.
               "network_calls")
TOP_FIELDS = ("ok", "code", "error", "errors", "warnings", "finished", "terminated",
              "submitted", "verified")
PROGRESS_FIELDS = ("completed", "total", "done", "status", "terminated")
CURRENT_FIELDS = ("id", "objective", "kind", "status")
CONTRACT_FIELDS = ("reusable", "analyze_needed", "stale_risk", "evidence_fresh")
# Platform chrome (status/navigation bars) is never application content.
SYSTEM_CHROME_PREFIXES = ("com.android.systemui:", "android:id/statusBarBackground",
                          "android:id/navigationBarBackground")


def _label(element: dict[str, Any]) -> str | None:
    for key in ("text", "desc", "content_desc"):
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _editable(element: dict[str, Any]) -> bool:
    if isinstance(element.get("editable"), bool):
        return element["editable"] is True
    # AUA's normalized hierarchy exposes a type, not necessarily an editable flag.
    # Never infer this from a label: a keyboard letter and an attachment button are not fields.
    kind = str(element.get("type") or "").rsplit(".", 1)[-1]
    return kind in {"EditText", "TextField", "SecureTextField", "TextArea"}


def _interactive(element: dict[str, Any]) -> bool:
    return _editable(element) or any(element.get(flag) is True for flag in STATE_FLAGS) or element.get("checked") is True


def _chrome(element: dict[str, Any]) -> bool:
    rid = element.get("resource_id") or element.get("rid") or ""
    return isinstance(rid, str) and rid.startswith(SYSTEM_CHROME_PREFIXES)


def _informative(element: dict[str, Any]) -> bool:
    if _chrome(element) and _label(element) is None and not _interactive(element):
        return False
    return (
        _label(element) is not None
        or bool(element.get("resource_id") or element.get("rid"))
        or _interactive(element)
        or element.get("selected") is True
    )


def _trim(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def compact_element(element: dict[str, Any], *, max_text: int = 120, keep_id: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ELEMENT_FIELDS:
        if key not in element:
            continue
        if key == "id" and not keep_id:
            continue
        value = element[key]
        # A false here really is the default and informs nothing, so it is dropped.
        if key in ("clickable", "editable", "selected", "scrollable", "focused") and value is not True:
            continue
        # `checked` is not like those: an off switch IS the reading, and dropping the false
        # makes it indistinguishable from an element that is no switch at all, which leaves
        # "the X switch is off" unprovable. A None does mean no switch, and still goes -- and
        # so does a raw hierarchy dump's `checked: false` on a node that says `checkable: false`
        # (every status-bar node has one), which is no reading either.
        if key == "checked" and (value is None or element.get("checkable") is False):
            continue
        if key in ("text", "desc", "content_desc", "resource_id", "rid") and (not isinstance(value, str) or not value.strip()):
            continue
        out[key] = _trim(value, max_text)
    if "content_desc" in out and "desc" not in out:
        out["desc"] = out.pop("content_desc")
    if _editable(element):
        out["editable"] = True
    return out


def _select(elements: list[dict[str, Any]], max_elements: int) -> tuple[list[dict[str, Any]], int]:
    kept = [(index, element) for index, element in enumerate(elements) if _informative(element)]
    dropped = len(elements) - len(kept)
    if len(kept) > max_elements:
        interactive = [pair for pair in kept if _interactive(pair[1])]
        passive = [pair for pair in kept if not _interactive(pair[1])]
        chosen = (interactive + passive)[:max_elements]
        dropped += len(kept) - len(chosen)
        kept = sorted(chosen, key=lambda pair: pair[0])
    return [element for _, element in kept], dropped


def _progress(progress: Any) -> Any:
    if not isinstance(progress, dict):
        return progress
    out = {key: progress[key] for key in PROGRESS_FIELDS if key in progress}
    current = progress.get("current")
    if isinstance(current, dict):
        out["current"] = {key: current[key] for key in CURRENT_FIELDS if key in current}
    upcoming = progress.get("upcoming")
    if isinstance(upcoming, list):
        out["upcoming"] = [
            {key: item[key] for key in ("id", "objective") if key in item}
            for item in upcoming if isinstance(item, dict)
        ][:4]
    return out


def _observation(result: dict[str, Any]) -> dict[str, Any] | None:
    nested = result.get("observation")
    if isinstance(nested, dict):
        return nested
    if isinstance(result.get("screen"), dict) and isinstance(result.get("elements"), list):
        return result
    # A refused action reports the screen under its error: the failure is about the action
    # AUA did not send, not about what is on display. Skipping this read made every such
    # step look blank, so the navigator declined it and the chat model paid for a frame it
    # could have answered itself.
    error = result.get("error")
    if isinstance(error, dict) and isinstance(error.get("observation"), dict):
        return error["observation"]
    return None


def compact_frame(
    result: Any,
    *,
    max_elements: int = 60,
    max_text: int = 120,
    previous_fingerprint: str | None = None,
    keep_ids: bool = True,
) -> Any:
    """Return a compact copy of one AUA result for a model prompt.

    Non-object results and results without an observation are returned unchanged apart
    from being copied. When ``previous_fingerprint`` matches the frame's fingerprint, the
    elements collapse to interactive handles so the model can still act without the
    full screen being resent.
    """
    if not isinstance(result, dict):
        return copy.deepcopy(result)
    observation = _observation(result)
    out: dict[str, Any] = {key: copy.deepcopy(result[key]) for key in TOP_FIELDS if key in result}
    if "goal_progress" in result:
        out["goal_progress"] = _progress(result["goal_progress"])
    contract = result.get("observation_contract")
    if isinstance(contract, dict):
        out["observation_contract"] = {key: contract[key] for key in CONTRACT_FIELDS if key in contract}
    if observation is None:
        for key, value in result.items():
            if key not in out and key not in ("observation_contract", "goal_progress"):
                out[key] = copy.deepcopy(value)
        return out
    screen = observation.get("screen") if isinstance(observation.get("screen"), dict) else {}
    meta = observation.get("meta") if isinstance(observation.get("meta"), dict) else {}
    elements = [e for e in observation.get("elements", []) if isinstance(e, dict)]
    fingerprint = meta.get("fingerprint")
    compact_meta = {key: meta[key] for key in META_FIELDS if key in meta}
    if "goal_progress" in meta and "goal_progress" not in out:
        out["goal_progress"] = _progress(meta["goal_progress"])
    frame: dict[str, Any] = {
        "screen": {key: screen[key] for key in ("package", "activity", "width", "height") if key in screen},
        "meta": compact_meta,
    }
    unchanged = (
        isinstance(fingerprint, str) and fingerprint == previous_fingerprint
        and not result.get("error") and result.get("ok") is not False
    )
    if unchanged:
        handles = [
            compact_element(e, max_text=40, keep_id=keep_ids)
            for e in elements if _interactive(e) and _informative(e)
        ][:max_elements]
        frame["unchanged"] = True
        frame["note"] = "Screen unchanged since the previous observation; interactive handles repeated, other elements omitted."
        frame["elements"] = handles
    else:
        selected, dropped = _select(elements, max_elements)
        frame["elements"] = [compact_element(e, max_text=max_text, keep_id=keep_ids) for e in selected]
        if dropped:
            frame["elided_elements"] = dropped
    out["observation"] = frame
    return out


class FrameCompactor:
    """Stateful compaction for one conversation: remembers the last fingerprint seen."""

    def __init__(self, *, max_elements: int = 60, max_text: int = 120, keep_ids: bool = True) -> None:
        self.max_elements = max_elements
        self.max_text = max_text
        self.keep_ids = keep_ids
        self.last_fingerprint: str | None = None
        self.frames_seen = 0
        self.unchanged_hits = 0

    def __call__(self, result: Any) -> Any:
        out = compact_frame(
            result, max_elements=self.max_elements, max_text=self.max_text,
            previous_fingerprint=self.last_fingerprint, keep_ids=self.keep_ids,
        )
        observation = out.get("observation") if isinstance(out, dict) else None
        if isinstance(observation, dict):
            self.frames_seen += 1
            if observation.get("unchanged"):
                self.unchanged_hits += 1
            fingerprint = (observation.get("meta") or {}).get("fingerprint")
            if isinstance(fingerprint, str):
                self.last_fingerprint = fingerprint
        return out
