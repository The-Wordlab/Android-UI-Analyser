"""Optional model-only width reduction for recognized native flow results.

This is not an evidence, freshness, or execution validator. The raw result, host
state and immutable history stay with the caller. Apply once to each new result.
"""

from __future__ import annotations

import copy
from typing import Any

ELEMENT_FIELDS = frozenset({
    "id", "type", "text", "resource_id", "rid", "content_desc", "desc",
    "clickable", "enabled", "focused",
})
_ALIASES = {"rid": "resource_id", "desc": "content_desc"}
_WRAPPERS = ("observation", "result", "data", "state", "structuredContent", "meta", "error")


def _full_frame(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    screen, elements, meta = value.get("screen"), value.get("elements"), value.get("meta")
    return (isinstance(screen, dict)
            and all(type(screen.get(key)) is int and screen[key] > 0 for key in ("width", "height"))
            and isinstance(elements, list)
            and all(isinstance(element, dict) and type(element.get("id")) in (str, int)
                    for element in elements)
            and isinstance(meta, dict) and isinstance(meta.get("fingerprint"), str)
            and bool(meta["fingerprint"].strip()))


def _frames(value: Any, depth: int = 0) -> list[dict]:
    if not isinstance(value, dict) or depth > 6:
        return []
    frames = [value] if _full_frame(value) else []
    for key in _WRAPPERS:
        frames.extend(_frames(value.get(key), depth + 1))
    return frames


def _canonical(element: dict) -> dict | None:
    result = {}
    for key, value in element.items():
        canonical = _ALIASES.get(key, key)
        if canonical in result and not _same(result[canonical], value):
            return None
        result[canonical] = value
    return result


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _duplicate(legacy: Any, full: list[dict]) -> bool:
    """Legacy renderings may omit defaults, but may not contradict or add facts."""
    if not isinstance(legacy, list) or len(legacy) != len(full):
        return False
    for old, current in zip(legacy, full, strict=True):
        if not isinstance(old, dict) or "id" not in old or not _same(old["id"], current["id"]):
            return False
        before, after = _canonical(old), _canonical(current)
        if before is None or after is None or any(
            key not in after or not _same(value, after[key]) for key, value in before.items()
        ):
            return False
    return True


def compact_flow_observations(value: Any) -> Any:
    """Copy only new native flow observations into the ordinary action field view.

    Recognize a flow outcome, not arbitrary element lists. Preserve every envelope,
    failure/step/resume field and observation metadata byte-for-byte in value. A
    second full frame is ambiguous and leaves the whole flow record unchanged.
    Failure or stale metadata remains explicit; trimming never makes it reusable.
    """
    result = copy.deepcopy(value)

    def visit(item: Any, depth: int = 0) -> None:
        if not isinstance(item, dict) or depth > 6:
            return
        native_flow = (type(item.get("ok")) is bool and isinstance(item.get("flow"), str)
                       and bool(item["flow"].strip()) and isinstance(item.get("steps_run"), list)
                       and item.get("observation_present") is True)
        if native_flow:
            frame = item.get("observation")
            if not _full_frame(frame) or len(_frames(item)) != 1:
                return
            elements = frame["elements"]
            if _duplicate(item.get("elements"), elements):
                del item["elements"]
            frame["elements"] = [
                {key: child for key, child in element.items() if key in ELEMENT_FIELDS}
                for element in elements
            ]
            return
        # Native route wrappers use result. Do not recurse into tool arguments,
        # steps, histories, arbitrary lists or serialized JSON strings.
        for key in ("result", "structuredContent"):
            visit(item.get(key), depth + 1)

    visit(result)
    return result
