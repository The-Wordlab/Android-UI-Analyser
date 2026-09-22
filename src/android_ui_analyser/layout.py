"""A screen's layout as a small tree: what is where, top to bottom.

``analyze`` returns a flat element list with bounds; the app map used to keep only the
names. This module turns that list into a tree an agent can read in one look. Containers
hold what sits inside them, repeated rows collapse into one line, system chrome and the
keyboard are dropped, and every label goes through the caller's redaction. The result is
rendered as text and stored once per screen record.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from .schema import Element

#: Framework/system resource namespaces that are never an app's own content.
_SYSTEM_ID_PREFIXES = ("com.android.systemui:", "android:")
#: Structural root ids that name nothing a reader cares about.
_GENERIC_CONTAINER_IDS = frozenset({"content", "action_bar_root", "container", "root"})
#: Upper bound on nodes kept per screen: enough for any real page, small enough to store.
MAX_NODES = 60

# Marks: tap · scroll · type · selected
TAP, SCROLL, INPUT, SELECTED = "◉", "↕", "✎", "★"


@dataclass
class LayoutNode:
    """One thing on the screen, with what sits inside it."""

    type: str
    label: str | None
    resource_id: str | None
    bounds: tuple[int, int, int, int]
    tap: bool = False
    scroll: bool = False
    input: bool = False
    selected: bool = False
    repeat: int = 1
    children: list[LayoutNode] = field(default_factory=list)

    @property
    def interesting(self) -> bool:
        """Does this node say anything a reader would miss if it were flattened away?"""
        return bool(
            self.label
            or (self.resource_id and self.resource_id not in _GENERIC_CONTAINER_IDS)
            or self.tap
            or self.scroll
            or self.input
            or self.selected
        )

    def _shape(self) -> tuple[str, str | None, bool, bool, bool, bool]:
        return (
            self.type,
            self.resource_id,
            self.label is None,
            self.tap,
            self.input,
            self.selected,
        )

    @property
    def _row_like(self) -> bool:
        """List rows share a resource id (or have no label); named controls never merge."""
        return bool(self.resource_id) or self.label is None


def _is_chrome(el: Element, height: int | None) -> bool:
    if el.window in {"system", "ime"}:
        return True
    if (el.resource_id or "").startswith(_SYSTEM_ID_PREFIXES):
        return True
    return height is not None and el.center[1] < 0.035 * height


def _is_input(el: Element) -> bool:
    t = (el.type or "").lower()
    return any(k in t for k in ("edittext", "textfield", "autocomplete", "searchview"))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
        and outer != inner
    )


def build_layout(
    elements: list[Element],
    *,
    height: int | None,
    label_of: Callable[[Element], str | None],
) -> list[LayoutNode]:
    """Nest the flat element list by bounds, drop chrome, flatten silence, collapse repeats.

    ``label_of`` is the caller's redaction (``memory.redact_label``): a typed value must
    never end up in the tree.
    """
    nodes: list[LayoutNode] = []
    stack: list[LayoutNode] = []
    for el in elements:
        if _is_chrome(el, height):
            continue
        bounds = (int(el.bounds[0]), int(el.bounds[1]), int(el.bounds[2]), int(el.bounds[3]))
        rid = el.resource_id.split("/")[-1].strip() if el.resource_id else None
        node = LayoutNode(
            type=(el.type or "View").rsplit(".", 1)[-1],
            label=label_of(el),
            resource_id=rid or None,
            bounds=bounds,
            tap=bool(el.clickable),
            scroll=el.scrollable is True,
            input=_is_input(el),
            selected=el.selected is True,
        )
        # a11y dumps are pre-order: the nearest earlier element whose box holds mine is
        # my parent. Pop what cannot contain me, then attach.
        while stack and not _contains(stack[-1].bounds, bounds):
            stack.pop()
        (stack[-1].children if stack else nodes).append(node)
        stack.append(node)
    trimmed = _collapse(_flatten(nodes))
    _cap(trimmed, [MAX_NODES])
    return trimmed


def _flatten(nodes: list[LayoutNode]) -> list[LayoutNode]:
    """Replace a node that says nothing with its children."""
    out: list[LayoutNode] = []
    for node in nodes:
        node.children = _flatten(node.children)
        if node.interesting:
            out.append(node)
        else:
            out.extend(node.children)
    return out


def _collapse(nodes: list[LayoutNode]) -> list[LayoutNode]:
    """Consecutive siblings of one shape become one line with a repeat count."""
    out: list[LayoutNode] = []
    for node in nodes:
        node.children = _collapse(node.children)
        if (
            out
            and node._row_like
            and out[-1]._shape() == node._shape()
            and not node.children
            and not out[-1].children
        ):
            out[-1].repeat += 1
        else:
            out.append(node)
    return out


def _cap(nodes: list[LayoutNode], budget: list[int]) -> None:
    """Keep at most ``budget`` nodes, depth-first, so the stored tree stays small."""
    kept: list[LayoutNode] = []
    for node in nodes:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        kept.append(node)
        _cap(node.children, budget)
    nodes[:] = kept


def render_layout(nodes: list[LayoutNode], *, title: str, height: int | None = None) -> str:
    """The tree as text. Bands: ``[top]``/``[bottom]`` of the screen, else ``@y``."""
    total = height or max((node.bounds[3] for node in _walk(nodes)), default=0)
    lines = [title]
    _emit(nodes, "", total, lines)
    return "\n".join(lines) + "\n"


def _walk(nodes: list[LayoutNode]) -> Iterator[LayoutNode]:
    for node in nodes:
        yield node
        yield from _walk(node.children)


def _emit(nodes: list[LayoutNode], prefix: str, total: int, lines: list[str]) -> None:
    for index, node in enumerate(nodes):
        last = index == len(nodes) - 1
        marks = "".join(
            mark
            for mark, on in (
                (TAP, node.tap),
                (SCROLL, node.scroll),
                (INPUT, node.input),
                (SELECTED, node.selected),
            )
            if on
        )
        name = (
            " ".join(
                part
                for part in (
                    node.resource_id if node.resource_id not in _GENERIC_CONTAINER_IDS else None,
                    f'"{node.label}"' if node.label else None,
                )
                if part
            )
            or node.type
        )
        x0, y0, x1, y1 = node.bounds
        if total and y1 <= 0.14 * total:
            where = "[top]"
        elif total and y0 >= 0.86 * total:
            where = "[bottom]"
        else:
            where = f"@y{y0}"
        repeat = f"  ×{node.repeat} similar" if node.repeat > 1 else ""
        lines.append(
            f"{prefix}{'└─ ' if last else '├─ '}{marks}{' ' if marks else ''}{name} {where} "
            f"{x1 - x0}×{y1 - y0}{repeat}"
        )
        _emit(node.children, prefix + ("   " if last else "│  "), total, lines)
