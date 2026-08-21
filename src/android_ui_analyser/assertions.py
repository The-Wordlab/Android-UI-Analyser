"""Platform-neutral structural assertions over normalized UI elements.

The evaluator deliberately consumes :class:`Element` objects rather than native hierarchy
formats.  Platform adapters own tree capture/normalization; this module only reasons about the
canonical parent links and traversal order they return.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import SelectorAmbiguousError, UsageError
from .schema import Element, ElementId, Source
from .selectors import element_digest, match_selector, selector_label

if TYPE_CHECKING:
    from .memory import RouteStep

Selector = dict[str, Any]


@dataclass(frozen=True)
class StructuralResult:
    """Matches remaining after structural predicates, or an explicit failure."""

    matches: tuple[Element, ...]
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.detail is None


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    detail: str


@dataclass(frozen=True)
class AssertionResult:
    """Result of evaluating one authored assertion against one settled frame."""

    ok: bool
    detail: str


def normalize_selector(value: Any, *, field: str = "selector") -> Selector:
    """Validate a nested selector mapping and normalize ``id`` to ``rid``.

    Nested relationship selectors intentionally have a smaller vocabulary than action
    selectors: one stable selector and an optional zero-based index.  They never guess a first
    match when the index is absent.
    """

    if not isinstance(value, dict):
        raise UsageError(f"{field} must be a selector mapping")
    body = dict(value)
    if "id" in body and "rid" in body:
        raise UsageError(f"{field} accepts id or rid, not both")
    if "id" in body:
        body["rid"] = body.pop("id")
    unknown = set(body) - {"rid", "text", "desc", "index"}
    if unknown:
        raise UsageError(f"{field} has unknown keys: {', '.join(sorted(unknown))}")
    chosen = [key for key in ("rid", "text", "desc") if body.get(key) is not None]
    if len(chosen) != 1:
        raise UsageError(f"{field} needs exactly one of id/rid/text/desc")
    key = chosen[0]
    value_text = body[key]
    if not isinstance(value_text, str) or not value_text.strip():
        raise UsageError(f"{field}.{key} must be a non-empty string")
    out: Selector = {key: value_text}
    if "index" in body:
        index = body["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise UsageError(f"{field}.index must be a non-negative integer")
        out["index"] = index
    return out


def parse_selector_expression(value: str, *, field: str) -> Selector:
    """Parse the CLI's compact ``rid:value`` / ``text:value`` / ``desc:value`` spelling."""

    prefix, separator, needle = value.partition(":")
    if not separator or prefix not in {"id", "rid", "text", "desc"} or not needle.strip():
        raise UsageError(
            f"invalid {field} selector {value!r}",
            hint=f"Use {field} 'rid:container', 'text:Label', or 'desc:Description'.",
        )
    return {"rid" if prefix == "id" else prefix: needle}


def _indexed_match(
    elements: Sequence[Element], selector: Selector, *, role: str
) -> tuple[Element | None, str | None]:
    query = {key: selector.get(key) for key in ("rid", "text", "desc")}
    matches = match_selector(elements, **query)
    label = selector_label(query)
    index = selector.get("index")
    if index is not None:
        if index >= len(matches):
            return None, f"{role} {label} index {index} has {len(matches)} matches"
        return matches[index], None
    if not matches:
        return None, f"{role} {label} matched 0 elements"
    if len(matches) > 1:
        raise SelectorAmbiguousError(
            f"{role} {label} matches {len(matches)} elements — add index to disambiguate",
            hint="candidates: " + " | ".join(element_digest(el) for el in matches[:8]),
        )
    return matches[0], None


def _descends_from(
    element: Element, ancestor_id: ElementId, by_id: dict[ElementId, Element]
) -> bool | None:
    """True/False for a known tree relation; None when structural evidence is unavailable."""

    if element.parent is None:
        return None if element.source is not Source.hierarchy else False
    visited: set[ElementId] = set()
    cursor: ElementId | None = element.parent
    while cursor is not None:
        if cursor == ancestor_id:
            return True
        if cursor in visited:
            return None
        visited.add(cursor)
        parent = by_id.get(cursor)
        if parent is None:
            return None
        cursor = parent.parent
    return False


def apply_structural_filters(
    elements: Sequence[Element],
    matches: Sequence[Element],
    *,
    within: Selector | None = None,
    same_parent_as: Selector | None = None,
) -> StructuralResult:
    """Filter subject matches with exact canonical-tree relationships.

    Bounds are never consulted.  If the normalized source cannot provide a parent relation,
    the assertion reports ``structural_evidence_unavailable`` rather than turning geometry into
    an invented hierarchy.
    """

    current = list(matches)
    by_id = {element.id: element for element in elements}
    if within is not None:
        container, failure = _indexed_match(elements, within, role="within")
        if failure:
            return StructuralResult((), failure)
        assert container is not None
        filtered: list[Element] = []
        unavailable = False
        for element in current:
            relation = _descends_from(element, container.id, by_id)
            if relation is True:
                filtered.append(element)
            elif relation is None:
                unavailable = True
        if not filtered and unavailable:
            return StructuralResult(
                (),
                "structural_evidence_unavailable: subject has no canonical ancestry for "
                f"within {selector_label(within)}",
            )
        current = filtered

    if same_parent_as is not None:
        peer, failure = _indexed_match(elements, same_parent_as, role="same_parent_as")
        if failure:
            return StructuralResult((), failure)
        assert peer is not None
        if peer.parent is None:
            return StructuralResult(
                (),
                "structural_evidence_unavailable: same_parent_as peer has no canonical parent",
            )
        filtered = [element for element in current if element.parent == peer.parent]
        if not filtered and any(element.parent is None for element in current):
            return StructuralResult(
                (),
                "structural_evidence_unavailable: subject has no canonical parent for "
                "same_parent_as",
            )
        current = filtered

    return StructuralResult(tuple(current))


def check_contains_all(
    elements: Sequence[Element], container: Element, selectors: Sequence[Selector]
) -> tuple[bool, str]:
    """Verify that every requested selector has a descendant of *container*."""

    by_id = {element.id: element for element in elements}
    missing: list[str] = []
    unavailable: list[str] = []
    for selector in selectors:
        query = {key: selector.get(key) for key in ("rid", "text", "desc")}
        candidates = match_selector(elements, **query)
        relations = [_descends_from(element, container.id, by_id) for element in candidates]
        descendants = [
            element
            for element, relation in zip(candidates, relations, strict=True)
            if relation is True
        ]
        index = selector.get("index")
        if index is not None:
            if index >= len(descendants):
                if any(relation is None for relation in relations):
                    unavailable.append(selector_label(query))
                    continue
                missing.append(
                    f"{selector_label(query)} index {index} "
                    f"({len(descendants)} descendant matches)"
                )
                continue
            descendants = [descendants[index]]
        if descendants:
            continue
        label = selector_label(query)
        if any(relation is None for relation in relations):
            unavailable.append(label)
        else:
            missing.append(label)
    if unavailable:
        return (
            False,
            "structural_evidence_unavailable: no canonical ancestry for "
            + ", ".join(unavailable),
        )
    if missing:
        return False, "contains_all missing descendants: " + ", ".join(missing)
    return True, "contains_all=" + ",".join(selector_label(selector) for selector in selectors)


def evaluate_order(
    elements: Sequence[Element], *, axis: str, selectors: Sequence[Selector]
) -> OrderResult:
    """Evaluate geometric order or normalized structural traversal (``reading``) order."""

    if axis not in {"horizontal", "vertical", "reading"}:
        return OrderResult(False, f"invalid order axis={axis!r}")
    located: list[Element] = []
    for position, selector in enumerate(selectors):
        element, failure = _indexed_match(elements, selector, role=f"selector[{position}]")
        if failure:
            return OrderResult(False, failure)
        assert element is not None
        located.append(element)
    if axis == "reading":
        traversal = {id(element): position for position, element in enumerate(elements)}
        values = [traversal[id(element)] for element in located]
        label = "positions"
    else:
        coordinate = 0 if axis == "horizontal" else 1
        values = [element.center[coordinate] for element in located]
        label = "centers"
    if all(left < right for left, right in zip(values, values[1:], strict=False)):
        return OrderResult(True, f"pass order axis={axis} {label}={values}")
    return OrderResult(False, f"expected strictly increasing {axis} {label}, got {values}")


def evaluate_assertion_step(step: RouteStep, elements: Sequence[Element]) -> AssertionResult:
    """Evaluate a parsed flow assertion without another device read.

    Session contracts call this with the exact :class:`AnalyzeResult` returned to the agent.
    That keeps proof tied to one settled frame and avoids the timing ambiguity of issuing one
    native hierarchy dump per assertion.
    """

    if step.kind == "assert-order":
        axis = step.assertion.get("axis")
        selectors = step.assertion.get("selectors")
        if axis not in {"horizontal", "vertical", "reading"} or not isinstance(selectors, list):
            return AssertionResult(False, "invalid assert_order payload")
        order = evaluate_order(elements, axis=axis, selectors=selectors)
        return AssertionResult(order.ok, order.detail)
    if step.kind != "assert":
        return AssertionResult(False, f"unsupported contract assertion kind={step.kind!r}")

    selector: Selector = {
        "rid": step.resource_id,
        "text": step.label,
        "desc": step.content_desc,
    }
    selector = {key: value for key, value in selector.items() if value is not None}
    if len(selector) != 1:
        return AssertionResult(False, "assertion needs exactly one subject selector")

    predicates = dict(step.assertion)
    first = bool(predicates.pop("first", False))
    count = predicates.pop("count", None)
    within_raw = predicates.pop("within", None)
    same_parent_raw = predicates.pop("same_parent_as", None)
    contains_raw = predicates.pop("contains_all", None)
    within = normalize_selector(within_raw, field="within") if within_raw is not None else None
    same_parent = (
        normalize_selector(same_parent_raw, field="same_parent_as")
        if same_parent_raw is not None
        else None
    )
    contains = tuple(
        normalize_selector(value, field=f"contains_all[{position}]")
        for position, value in enumerate(contains_raw or ())
    )
    label = selector_label(selector)
    matches = match_selector(elements, **selector)
    structural = apply_structural_filters(
        elements,
        matches,
        within=within,
        same_parent_as=same_parent,
    )
    if not structural.ok:
        return AssertionResult(False, f"{label}: {structural.detail}")
    matches = list(structural.matches)
    if count is not None and len(matches) != count:
        return AssertionResult(False, f"{label}: expected count={count}, actual={len(matches)}")
    if count == 0:
        return AssertionResult(True, f"{label}: count=0")
    if predicates.get("absent"):
        return AssertionResult(not matches, f"{label}: {'absent' if not matches else 'present'}")
    if not matches:
        return AssertionResult(False, f"{label}: absent")

    state_predicates = [name for name in predicates if name not in {"exists", "absent"}]
    if len(matches) > 1 and (state_predicates or within or same_parent or contains) and not (
        step.index is not None or first
    ):
        raise SelectorAmbiguousError(
            f"{label} matches {len(matches)} elements — add index or first to disambiguate",
            hint="candidates: " + " | ".join(element_digest(el) for el in matches[:8]),
        )
    if step.index is not None and step.index >= len(matches):
        return AssertionResult(
            False,
            f"{label}: index {step.index} out of range for {len(matches)} matches",
        )
    element = matches[step.index] if step.index is not None else matches[0]
    if contains:
        ok, detail = check_contains_all(elements, element, contains)
        if not ok:
            return AssertionResult(False, f"{label}: {detail}")

    labels = [value for value in (element.text, element.content_desc) if value]
    failures: list[str] = []
    for name, wanted in predicates.items():
        if name in {"exists", "absent"}:
            continue
        if name == "text_is":
            if not any(value.strip() == wanted for value in labels):
                failures.append(f"text_is={wanted!r} actual={labels or None!r}")
        elif name == "text_contains":
            if not any(str(wanted).lower() in value.lower() for value in labels):
                failures.append(f"text_contains={wanted!r} actual={labels or None!r}")
        elif bool(getattr(element, name, None)) is not bool(wanted):
            failures.append(f"{name}={wanted!r} actual={getattr(element, name, None)!r}")
    if failures:
        return AssertionResult(False, f"{label}: " + "; ".join(failures))
    return AssertionResult(True, f"{label}: " + (",".join(predicates) or "exists"))
