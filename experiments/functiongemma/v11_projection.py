"""The compact on-device projection: ~41 real nodes down to the handful that carry a decision.

Measured against **638 screens harvested from live emulators** across stock AOSP (319), third-party
and locally-installed apps (208), and the same screens under English, Arabic and German (111).

| property, clickable nodes | combined | AOSP | apps | locale set |
|---|---:|---:|---:|---:|
| carries visible text | **61.8%** | 60.4% | 64.5% | 60.0% |
| carries a description | 32.0% | 39.3% | 31.5% | 47.9% |
| carries a resource id | 34.7% | 44.2% | 31.7% | 30.6% |
| **neither text nor description** | **11.2%** | 8.0% | 6.9% | 1.6% |

Median nodes per screen: 41 combined, 48 on AOSP, 17 in third-party apps.

**Read the spread, not the average.** An earlier draft of this docstring quoted 29% text and 15.5%
unnameable as settled fact; both came from generalising a single Settings screen and one harvest, and
both were wrong — text is the *common* field, not the rare one. The per-corpus columns differ by a
factor of five on the unnameable rate, so it is a property of which apps were sampled, not of Android.
Open-source and preference-heavy screens are unusually well labelled; games are not.

**The failure this table cannot see.** Drawn surfaces do not produce badly-named nodes. They produce
**one node, or none**:

* a live backgammon board reports 10 nodes, 6 clickable and *all 6 named* — every one of them
  left-column chrome (Settings, Resign, New game, Home). The board contributes zero: no dice, no
  points, no checkers, no Undo, no Commit. An agent can start a match and resign it; it cannot play.
* a 2048 implementation reports **zero nodes** and its intro modal cannot be dismissed at all
* a chess board is one node, `rid:chessboard` — while `rid:status` ("1. White's move") reads fine, so
  the *state* is legible while no *move* is addressable
* a sudoku grid is one node and not even clickable, though the 1-9 entry buttons are named

A naming ratio cannot be computed over nodes that do not exist, so no amount of selector policy helps
here. The correct answer on such a screen is to hand off, and the projection's job is to make that
visible rather than to pretend there is a choice.

**Resource ids survive a locale switch; text does not.** Across matched screens the clickable
resource-id set is identical between English and Arabic, while the text set shares almost nothing
(Jaccard 0.026 for Arabic, 0.042 for German) — and no clickable node *lost* a label, 96-99% of them
merely changed value. That argues for preferring the id. The catch is coverage: only ~26% of clickable
nodes carry a resource id *themselves* (in AOSP the clickable row container is unnamed and the id sits
on a child), ~53% are reachable through one somewhere in their subtree, and the remaining ~47% can be
named only by the text or description that the locale changes. AUA's own composite `id` is *less*
stable than a raw resource id across locales (~0.65-0.72), because it falls back to text-derived keys
wherever no id exists.

What this projection does, in order:

1. **Drop what can neither be acted on nor read.** Not keyed on a list of known container ids — the
   first version was, and it leaked `status_bar_launch_animation_container` and every other wrapper
   nobody had thought to list.
2. **Drop status-bar and shade content** that survives that filter by carrying text: the clock, the
   battery, notification summaries. Other applications' notifications have no business in a corpus.
3. **Collapse duplicates onto the actionable twin.** A non-clickable heading repeating a clickable
   row's text is ordinary Android layout, not a puzzle, and resolving it here deletes a whole
   synthetic training family instead of teaching it.
4. **Ordinals, never hashes.** ``tx:8da862873d`` is ten hex characters a model must reproduce
   exactly; at 4-bit that is copy poison. Nodes are numbered ``n1..nN`` and the caller keeps the map
   back to AUA's stable key.
5. **Cap context at two non-actionable nodes**, preferring text over description, so a status bar can
   never crowd out a control and the slots hold the screen's heading.
6. **Normalise** — NFKC, non-breaking punctuation to ASCII, collapsed whitespace. Real screens carry
   U+2011 in "Wi-Fi" and a small model should not spend capacity on it.
7. **Truncate honestly.** Above ``MAX_NODES`` the projection sets ``more`` — and that flag changes
   what the correct answer *is*: with content off-screen, "the target is absent" is not provable and
   scrolling is the only sound step. A corpus that truncates silently teaches "not in the list,
   therefore refuse", which is wrong on any scrolling screen.

One data-hygiene note for whoever consumes the harvest: the three files do not agree on element key
names. ``aosp.jsonl`` and ``apps.jsonl`` use ``rid``/``desc``; ``locale.jsonl`` uses
``resource_id``/``content_desc``. Reading the wrong pair silently reports 0% coverage, which is
exactly the false "40% of Arabic controls are unnameable" reading it produced before being caught.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: Nodes shown to the model. Real screens have a median of 6 actionable nodes, so this leaves
#: headroom for dense ones while keeping the prompt small enough for an on-device budget.
MAX_NODES = 14

#: Windows that are never the app under test. `analyze` folds them into one tree.
FOREIGN_PACKAGES = ("com.android.systemui",)

#: Status-bar and shade content that carries text or a description and therefore survives the
#: structural filter: the clock, the battery, signal strength, notification summaries. A driver
#: deciding what to tap gains nothing from "Battery charging, 100 percent.", and on a dense screen
#: these would consume the node budget. Notification text is also other applications' content,
#: which has no business in a training corpus.
#:
#: Matched on the resource-id tail, and these are stable AOSP systemui ids. Text-only status nodes
#: with no id are caught instead by the context cap below.
SYSTEMUI_RID_PREFIXES = (
    "status_bar",
    "statusIcons",
    "notification",
    "clock",
    "battery",
    "wifi_",
    "mobile_",
    "system_icons",
    "shade",
    "qs_",
    "keyguard",
)

#: Non-actionable nodes kept as context — normally the screen's title. Capping this is what stops
#: the status bar from filling the budget: actionable nodes are ranked first and never displaced,
#: and a driver needs a heading for orientation, not the whole readable surface.
MAX_CONTEXT_NODES = 2

_WS = re.compile(r"\s+")
#: Non-breaking and typographic punctuation real screens use. "Wi‑Fi" is U+2011, not a hyphen.
_PUNCT = {
    "‑": "-",
    "‐": "-",
    "–": "-",
    "—": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    " ": " ",
    "​": "",
}


def normalise(value: Any) -> str | None:
    """NFKC, ASCII punctuation, collapsed whitespace. ``None`` for anything empty."""

    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    for bad, good in _PUNCT.items():
        text = text.replace(bad, good)
    text = _WS.sub(" ", text).strip()
    return text or None


def _rid_tail(rid: Any) -> str | None:
    if not rid:
        return None
    tail = str(rid).rsplit("/", 1)[-1]
    return tail.split("#", 1)[0] or None


def is_structural(node: Mapping[str, Any]) -> bool:
    """A node that can neither be acted on nor read, whatever its resource id says.

    Deliberately *not* keyed on a list of known structural resource ids. The first version was, and
    on real screens it leaked `status_bar_launch_animation_container`,
    `status_bar_contents` and friends straight into the model's view — the enumerated list can only
    ever cover the containers someone thought of. The property that matters is simpler and total: a
    node that is not clickable, not scrollable, and carries no text and no description offers the
    model nothing to choose and nothing to verify. Its resource id is irrelevant, because a driver
    is never asked to point at scaffolding.

    Note this also does the window filtering that `FOREIGN_PACKAGES` cannot: the status bar and
    notification shade reach us as unnamed containers, and `analyze` does not stamp a per-node
    package for us to filter on. The nodes from those windows that *do* carry text (a clock, a
    battery description) survive, which is correct — they are readable screen state.
    """

    if node.get("clickable") or node.get("scrollable"):
        return False
    return not normalise(node.get("text")) and not normalise(node.get("desc"))


def project(
    elements: Sequence[Mapping[str, Any]],
    *,
    package: str | None = None,
    max_nodes: int = MAX_NODES,
) -> dict[str, Any]:
    """Reduce a raw ``analyze`` element list to the compact view a driver is shown.

    Returns ``{"nodes": [...], "more": bool, "keys": [...]}`` where ``keys[i]`` is AUA's stable key
    for ``nodes[i]``, kept by the caller and never shown to the model.
    """

    # 1. foreign windows
    kept: list[Mapping[str, Any]] = []
    for node in elements:
        node_package = node.get("package")
        if node_package and node_package in FOREIGN_PACKAGES:
            continue
        if package and node_package and node_package != package:
            continue
        kept.append(node)

    # 2. structure
    kept = [node for node in kept if not is_structural(node)]

    # 3. duplicate collapse — a non-actionable node repeating an actionable one's text adds nothing
    actionable_text = {
        normalise(node.get("text"))
        for node in kept
        if node.get("clickable") and normalise(node.get("text"))
    }
    collapsed: list[Mapping[str, Any]] = []
    for node in kept:
        text = normalise(node.get("text"))
        if not node.get("clickable") and text and text in actionable_text:
            continue
        collapsed.append(node)

    # 4. rank: actionable first, then anything readable. Order within a rank is preserved so the
    #    projection still reflects the screen's own layout order.
    def rank(node: Mapping[str, Any]) -> int:
        if node.get("clickable"):
            return 0
        if node.get("scrollable"):
            return 1
        return 2

    collapsed = [
        node
        for node in collapsed
        if not (
            (tail := _rid_tail(node.get("rid")))
            and tail.startswith(SYSTEMUI_RID_PREFIXES)
            and not node.get("clickable")
        )
    ]

    actionable = [node for node in collapsed if node.get("clickable") or node.get("scrollable")]
    context = [node for node in collapsed if node not in actionable]
    # A screen's heading carries *text*; a notification summary carries only a *description*, and
    # those arrive with no resource id, so the systemui filter above cannot see them. Preferring
    # text keeps the two context slots for orientation rather than for other apps' notifications,
    # while tree order breaks ties so the heading beats a footer.
    context.sort(key=lambda node: (0 if normalise(node.get("text")) else 1,))
    # Actionable nodes are never displaced by context, and context is capped hard so a status bar
    # cannot crowd out a control. `more` therefore reports genuinely dense *actionable* screens.
    more = len(actionable) > max_nodes
    ordered = actionable[:max_nodes]
    room = max_nodes - len(ordered)
    if room > 0:
        ordered = ordered + context[: min(room, MAX_CONTEXT_NODES)]

    nodes: list[dict[str, Any]] = []
    keys: list[str | None] = []
    for index, node in enumerate(ordered, start=1):
        out: dict[str, Any] = {"n": f"n{index}"}
        text = normalise(node.get("text"))
        desc = normalise(node.get("desc"))
        tail = _rid_tail(node.get("rid"))
        if text:
            out["text"] = text
        if desc and desc != text:
            out["desc"] = desc
        # A resource id is shown only when it is the *only* thing naming the node: 15.5% of real
        # clickable controls have neither text nor description, and for those the id is the sole
        # readable evidence. Showing it everywhere would reintroduce the `navTab`-style leak where
        # the id is just the label in another font.
        if tail and not text and not desc:
            out["rid"] = tail
        if node.get("clickable"):
            out["tap"] = True
        if node.get("scrollable"):
            out["scroll"] = True
        nodes.append(out)
        keys.append(node.get("id"))

    return {"nodes": nodes, "more": more, "keys": keys}


def unnameable_rate(elements: Iterable[Mapping[str, Any]]) -> float:
    """Share of clickable nodes with no text and no description — addressable only by position."""

    clickable = [node for node in elements if node.get("clickable")]
    if not clickable:
        return 0.0
    blind = sum(
        1
        for node in clickable
        if not normalise(node.get("text")) and not normalise(node.get("desc"))
    )
    return blind / len(clickable)
