"""V11 source material: on-device navigation trajectories over fictional app graphs.

Every earlier cycle in this experiment taught a single isolated decision. V11 teaches a *journey*,
because that is what the goal actually is: given "open the app, go to the apps section, land on
explore", drive until the destination is proven. A journey is expanded into one training row per
state along it, so the model learns "what do I do from *here*" at every point, including the last
one where the right answer is to stop.

Three design rules come straight out of this experiment's recorded failures, and breaking any of
them reproduces a defect that already cost a cycle.

**A family is not taught by its own examples alone — it is taught by what else varies while it is
being taught.** V9 confounded candidate count with the label and action direction with the family.
V10 confounded refusal with destructiveness: 88.2% of its handoff cases contained an unauthorized
or destructive candidate, so the cheapest consistent rule was "refuse when something dangerous is
on the menu", decidable from flags without reading the goal at all. Here, destructive controls are
sprinkled at the *same rate* through acting and refusing states (see ``DESTRUCTIVE_RATE``), refusal
is dominated by relevance rather than authorization (``RELEVANCE_REFUSAL_SHARE``), and node counts
are drawn independently of the label.

**A probe that shares the generator's phrasing measures that phrasing.** V9 scored 6/6 in-repo on
refusing an absent target and 0/144 under an independently authored audit. Goals here are rendered
through many templates (``GOAL_TEMPLATES``), and the templates are partitioned by split so a
held-out goal is phrased in a way training never used.

**The selector is the answer, not an index.** The model is never shown a node index, so the only
way to name a control is to author a real selector. Controls deliberately vary in what they expose:
some carry only a resource id, some only a content description. The V9 live incident — an absent
destination tapped because a rid-only control could not be read — is exactly the case a driver has
to handle, so ``rid_only`` and ``desc_only`` controls appear throughout rather than in one family.

Everything here is fictional and app-agnostic, as the public-corpus boundary requires. Screen
names, control labels, resource ids and packages are drawn from invented vocabularies, partitioned
by split so no semantic group can appear on both sides of a split boundary.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .v11_contract import HOST_ONLY_CAPABILITIES

SCHEMA = "aua-v11-device-driver-trajectories"
SEED = 111

#: Share of states that carry a destructive control somewhere on screen. Held constant across
#: acting and refusing states so "something dangerous is visible" carries no information about
#: whether the right answer is to refuse. This is the direct fix for the V10 confound.
DESTRUCTIVE_RATE = 0.30

#: Share of handoff states whose refusal is a *relevance* judgement — every visible control is
#: safe and authorized, and none of them advances the goal. V10 had this at 11.8% and the model
#: duly learned the other rule. The two authorization-shaped refusals are the minority here.
RELEVANCE_REFUSAL_SHARE = 0.80

#: Maximum steps a driver is given before the run is considered stuck.
DEFAULT_BUDGET = 10


# --------------------------------------------------------------------------- vocabularies

# Split-disjoint pools. A destination name learned in `train` can never be scored in `test`, so a
# correct answer has to come from reading the goal against the screen rather than from recall.
_SECTION_POOLS: dict[str, tuple[str, ...]] = {
    "train": (
        "Explore",
        "Library",
        "Digest",
        "Almanac",
        "Roster",
        "Ledger",
        "Atlas",
        "Bulletin",
        "Compendium",
        "Gazette",
        "Registry",
        "Manifest",
        "Directory",
        "Archive",
        "Journal",
        "Catalog",
        "Index",
        "Anthology",
        "Chronicle",
        "Dossier",
        "Folio",
        "Gallery",
    ),
    "valid": (
        "Observatory",
        "Herbarium",
        "Vestibule",
        "Belfry",
        "Pantry",
        "Cloister",
        "Refectory",
        "Undercroft",
        "Lyceum",
        "Vivarium",
        "Apiary",
        "Cellarium",
    ),
    "test": (
        "Scriptorium",
        "Cartulary",
        "Conservatory",
        "Rotunda",
        "Solarium",
        "Orangery",
        "Campanile",
        "Triforium",
        "Narthex",
        "Chancery",
        "Sacristy",
        "Colonnade",
    ),
}

#: Sections that sound like each other. A near-miss distractor is drawn from the same pool as the
#: target so the model cannot separate them by vocabulary, only by exact match against the goal.
_NEAR_MISS_SUFFIXES = ("s", " Hub", " Home", " Beta", " (Legacy)", " Settings")

_APP_STEMS: dict[str, tuple[str, ...]] = {
    "train": ("atlas", "cobalt", "harbor", "juniper", "lantern", "meridian", "onyx", "quartz"),
    "valid": ("saffron", "tundra"),
    "test": ("verdigris", "zephyr"),
}

_GOAL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "train": (
        "Open the {dest} section and prove its page",
        "Navigate to {dest}",
        "Go to the {dest} screen",
        "Take me to {dest}",
        "I need the {dest} view",
        "Reach {dest} and confirm it is showing",
        "Show the {dest} page",
        "Land on {dest}",
    ),
    # Held-out phrasings. The V9 lesson: an adapter that memorised one template flipped from
    # 48/48 correct to 48/48 wrong when an article was inserted.
    "valid": (
        "Bring up {dest}",
        "Get to the {dest} area",
    ),
    "test": (
        "Could you open {dest} for me",
        "End up on the {dest} tab",
    ),
}

#: Multi-hop goals name the whole route, because an intermediate hop is otherwise unlearnable:
#: nothing on a home screen reveals which section contains the destination. This mirrors how the
#: goal is actually phrased in practice — "navigate to the apps section and land on explore" names
#: the waypoint and the destination. A goal that named only the destination would be asking the
#: model to guess an app's information architecture from one screen.
_ROUTE_TEMPLATES: dict[str, dict[int, tuple[str, ...]]] = {
    "train": {
        2: (
            "Open {h0}, then land on {h1}",
            "Go to {h0} and from there reach {h1}",
            "Navigate to the {h0} section and land on {h1}",
            "Take me through {h0} to {h1}",
        ),
        3: (
            "Open {h0}, then {h1}, then land on {h2}",
            "Go to {h0}, into {h1}, and finish on {h2}",
            "Navigate {h0} to {h1} to {h2}",
        ),
    },
    "valid": {
        2: ("Work through {h0} until you are on {h1}",),
        3: ("Follow {h0} then {h1} and stop on {h2}",),
    },
    "test": {
        2: ("Head into {h0} and end up at {h1}",),
        3: ("Route via {h0} and {h1}, arriving at {h2}",),
    },
}


#: Host-lane goals, partitioned by split. Each entry asks for the same capability in wording the
#: other splits never use, so recognising "this needs the host" cannot be template recall.
_HOST_LANE_GOALS: dict[str, dict[str, tuple[str, ...]]] = {
    "proxy": {
        "train": (
            "Capture the network calls this screen makes",
            "Record the HTTP traffic behind this list",
        ),
        "valid": ("Intercept what this page requests",),
        "test": ("Trace the outbound requests from here",),
    },
    "database": {
        "train": (
            "Read the app's local database and report the row count",
            "Query the stored records for this section",
        ),
        "valid": ("Inspect the on-disk tables behind this view",),
        "test": ("Report what the local store holds for this page",),
    },
    "install": {
        "train": (
            "Install the new build and relaunch it",
            "Put the updated package on the target and open it",
        ),
        "valid": ("Side-load the candidate build, then restart",),
        "test": ("Deploy the fresh artifact and bring it up",),
    },
    "emulator": {
        "train": (
            "Boot a second emulator and mirror this screen",
            "Start another virtual target for comparison",
        ),
        "valid": ("Bring up an extra device image alongside this one",),
        "test": ("Spin up a parallel target and compare",),
    },
    "flags": {
        "train": (
            "Turn on the experimental layout feature flag",
            "Enable the gated variant for this screen",
        ),
        "valid": ("Switch the trial toggle on for this build",),
        "test": ("Activate the withheld configuration option",),
    },
    "logcat": {
        "train": (
            "Collect the app's logs for this screen",
            "Gather the diagnostic output around this view",
        ),
        "valid": ("Pull the runtime log lines for this page",),
        "test": ("Retrieve what the process printed here",),
    },
    "clipboard_host": {
        "train": (
            "Put the host clipboard contents into this field",
            "Paste what the workstation copied into this input",
        ),
        "valid": ("Transfer the desktop selection into this box",),
        "test": ("Move the machine's copied text into this field",),
    },
    "network_profile": {
        "train": (
            "Re-run this screen on a slow network profile",
            "Repeat this view under a degraded connection",
        ),
        "valid": ("Try this page on a throttled link",),
        "test": ("Replay this screen with constrained bandwidth",),
    },
}

#: Search and authorization goals, likewise partitioned. ``{dest}`` is filled from the split's own
#: destination pool, so both the verb phrasing and the noun are held out together.
_SEARCH_TEMPLATES: dict[str, tuple[str, ...]] = {
    "train": (
        "Search for {dest} and open the first result",
        "Look up {dest} and pick what comes back",
    ),
    "valid": ("Find {dest} through the search field",),
    "test": ("Query for {dest} and take the top hit",),
}

_AUTHORIZATION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "train": ("Remove the {dest} entry", "Delete the {dest} record"),
    "valid": ("Erase the {dest} item",),
    "test": ("Discard the stored {dest}",),
}

_DESTRUCTIVE_LABELS = (
    "Delete account",
    "Erase all data",
    "Reset everything",
    "Sign out",
    "Remove downloads",
    "Clear history",
    "Wipe cache",
    "Revoke access",
)

_INERT_LABELS = (
    "Version",
    "Build number",
    "Last synced",
    "Storage used",
    "Legal notices",
    "Open-source licences",
    "Region",
    "Language",
)


#: Resource-id shapes real Android screens use. The first draft emitted `navTab` + the camel-cased
#: section for **every** rid in **all three splits**, which meant a resource id was never an
#: identifier to be read — it was the label in a different font, and the split partition could not
#: detect the leak. A trained model duly invented `navTabBulletin` on screens containing no such
#: node. Styles are chosen per control from a stable hash so a trajectory stays self-consistent.
_RID_STYLES = ("nav", "camel", "snake", "opaque", "prefixed")
_RID_PREFIXES = ("btn", "row", "cell", "fab", "v", "ctl", "item")


def _rid(section: str, prefix: str = "navTab") -> str:
    """A resource-id tail in the shape Android apps actually use.

    ``prefix`` is retained for the two synthetic controls named by hand (`searchField`,
    `scroller`); section controls go through :func:`_styled_rid` instead.
    """

    return prefix + "".join(
        part.capitalize() for part in section.replace("(", "").replace(")", "").split()
    )


def _styled_rid(section: str) -> str:
    """One resource id for *section*, in a style deterministically derived from its name.

    Only the ``nav`` and ``camel`` styles contain the section's words at all. The rest are opaque,
    which is the case the V9 live incident was actually about: a control whose resource id is the
    only thing identifying it and whose id says nothing.
    """

    digest = hashlib.sha256(f"rid:{section}".encode()).hexdigest()
    style = _RID_STYLES[int(digest[:2], 16) % len(_RID_STYLES)]
    words = section.replace("(", "").replace(")", "").split()
    camel = "".join(part.capitalize() for part in words)
    if style == "nav":
        return "navTab" + camel
    if style == "camel":
        return camel[:1].lower() + camel[1:] + "Button"
    if style == "snake":
        return "_".join(word.lower() for word in words)
    if style == "opaque":
        return f"{digest[2:6]}_{digest[6:9]}"
    prefix = _RID_PREFIXES[int(digest[9:11], 16) % len(_RID_PREFIXES)]
    return f"{prefix}_{digest[11:15]}"


# --------------------------------------------------------------------------- screen model


@dataclass(frozen=True)
class Control:
    """One control on a fictional screen.

    ``exposes`` decides which of text / description / resource id the control actually carries.
    A real Compose screen frequently gives a control exactly one of the three, which is why a
    driver has to be able to name any of them.
    """

    section: str
    exposes: str  # "text" | "desc" | "rid_only" | "text_and_rid"
    clickable: bool = True
    destructive: bool = False
    inert: bool = False
    scrollable: bool = False
    offscreen: bool = False

    @property
    def rid(self) -> str | None:
        if self.exposes in ("rid_only", "text_and_rid"):
            return _styled_rid(self.section)
        return None

    @property
    def text(self) -> str | None:
        if self.exposes in ("text", "text_and_rid"):
            return self.section
        return None

    @property
    def desc(self) -> str | None:
        if self.exposes == "desc":
            return self.section
        return None

    def selector(self) -> dict[str, str]:
        """The one selector a driver should use for this control, in the helper's priority order."""

        if self.rid:
            return {"resource_id": self.rid}
        if self.text:
            return {"label": self.text}
        return {"content_desc": self.desc or self.section}

    def as_node(self) -> dict[str, Any]:
        """The compact projection the device sees. No bounds, no index — selectors only."""

        node: dict[str, Any] = {}
        if self.rid:
            node["rid"] = self.rid
        if self.text:
            node["text"] = self.text
        if self.desc:
            node["desc"] = self.desc
        if self.clickable:
            node["clickable"] = True
        if self.scrollable:
            node["scrollable"] = True
        return node


@dataclass
class Screen:
    """A fictional screen: a name, a title, and the controls on it."""

    name: str | None
    title: str | None
    controls: list[Control] = field(default_factory=list)
    loading: bool = False
    ime_up: bool = False

    def nodes(self, rnd: random.Random) -> list[dict[str, Any]]:
        visible = [control for control in self.controls if not control.offscreen]
        nodes = [control.as_node() for control in visible]
        if self.title:
            # A passive page title that duplicates a row's copy is the breadcrumb trap: the
            # correct answer is the clickable row, not the heading that reads the same.
            nodes.append({"text": self.title})
        rnd.shuffle(nodes)
        return nodes


# --------------------------------------------------------------------------- trajectory model


@dataclass
class State:
    """One decision point: what the driver sees, and the one right answer."""

    goal: str
    screen: Screen
    history: list[str]
    call: str  # "next_step" | "done" | "handoff"
    arguments: dict[str, Any]
    family: str
    budget: int = DEFAULT_BUDGET
    package: str = "example.package"
    known_screen: str | None = None
    host_lane_available: bool = False

    def as_context(self, rnd: random.Random) -> dict[str, Any]:
        """The user-turn payload. Kept small: this has to fit an on-device context window."""

        screen: dict[str, Any] = {"package": self.package, "nodes": self.screen.nodes(rnd)}
        # `known_screen` is deliberately NOT emitted. The helper's `ui.tree` returns a node list and
        # nothing else; a semantic screen name is a host-side AUA concept. Training on it would mean
        # leaning at inference on a field that is never there — and it leaked as well, letting
        # "no screen name plus a tap in the history" imply a stalled run.
        if self.screen.loading:
            screen["settling"] = True
        if self.screen.ime_up:
            screen["ime"] = True
        context: dict[str, Any] = {
            "goal": self.goal,
            "step": len(self.history),
            "budget": self.budget,
            "screen": screen,
        }
        if self.history:
            context["history"] = list(self.history)
        if self.host_lane_available:
            context["host_lane"] = True
        return context


def _describe(call: str, arguments: dict[str, Any]) -> str:
    """One history line, in the same vocabulary the model emits. Keeps the loop self-consistent."""

    if call != "next_step":
        return call
    kind = arguments["kind"]
    for key in ("resource_id", "label", "content_desc"):
        if arguments.get(key):
            short = {"resource_id": "rid", "label": "text", "content_desc": "desc"}[key]
            return f"{kind} {short}={arguments[key]}"
    if arguments.get("arg"):
        return f"{kind} {arguments['arg']}"
    return kind


# --------------------------------------------------------------------------- generators

_EXPOSURES = ("text", "text_and_rid", "rid_only", "desc")


def _tab_row(rnd: random.Random, sections: list[str], *, exposures: list[str]) -> list[Control]:
    return [
        Control(section=section, exposes=exposure)
        for section, exposure in zip(sections, exposures, strict=True)
    ]


def _maybe_destructive(rnd: random.Random) -> list[Control]:
    """A destructive control, at a rate independent of whether the answer is to refuse."""

    if rnd.random() >= DESTRUCTIVE_RATE:
        return []
    return [
        Control(
            section=rnd.choice(_DESTRUCTIVE_LABELS),
            exposes=rnd.choice(("text", "text_and_rid")),
            destructive=True,
        )
    ]


def _noise(rnd: random.Random, sections: list[str]) -> list[Control]:
    """Inert, non-clickable rows. Present so "clickable" is not a free giveaway."""

    count = rnd.randint(0, 2)
    return [
        Control(section=rnd.choice(_INERT_LABELS), exposes="text", clickable=False, inert=True)
        for _ in range(count)
    ]


def _goal_for(rnd: random.Random, split: str, dest: str) -> str:
    """One destination goal. A fifth are lower-cased, so capitalisation is not the signal.

    Real goals arrive as "open settings", not "Open the Settings". Keeping every destination
    Capitalised made "does this goal contain a capital?" separate device work from host work, and
    also let the model find the target by looking for the one rare capitalised word instead of by
    reading it.
    """

    if rnd.random() < 0.18:
        # Search-shaped phrasing for a task that is reached by tapping, not typing.
        goal = rnd.choice(
            (
                "Search the app for {dest} and open it",
                "Look for {dest} and go there",
                "Find {dest}",
            )
        ).format(dest=dest)
    else:
        goal = rnd.choice(_GOAL_TEMPLATES[split]).format(dest=dest)
    if rnd.random() < 0.20:
        goal = goal[:1] + goal[1:].lower()
    return goal + rnd.choice(
        (
            "",
            "",
            " please",
            " when you get a chance",
            " and report back",
            " before anything else",
            " — this is the whole task",
        )
    )


def _package(rnd: random.Random, split: str) -> str:
    return f"example.{rnd.choice(_APP_STEMS[split])}.app"


def _pick(rnd: random.Random, pool: tuple[str, ...], count: int) -> list[str]:
    return rnd.sample(list(pool), min(count, len(pool)))


def _pick_leaving_one(rnd: random.Random, pool: tuple[str, ...], count: int) -> list[str]:
    """Draw sections while guaranteeing at least one stays absent.

    The refusal families need a destination that is genuinely not on screen. Drawing the
    whole pool leaves none, which is a generator crash rather than a training row.
    """

    return _pick(rnd, pool, min(count, len(pool) - 1))


def _near_miss(rnd: random.Random, target: str) -> str:
    return target + rnd.choice(_NEAR_MISS_SUFFIXES)


def _states_direct(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """The destination is on screen now: tap it, prove it, stop."""

    pool = _SECTION_POOLS[split]
    sections = _pick(rnd, pool, rnd.randint(3, 5))
    dest = sections[0]
    goal = _goal_for(rnd, split, dest)
    package = _package(rnd, split)
    exposures = [rnd.choice(_EXPOSURES) for _ in sections]
    tabs = _tab_row(rnd, sections, exposures=exposures)
    target = tabs[0]

    home = Screen(
        name="home", title=None, controls=[*tabs, *_maybe_destructive(rnd), *_noise(rnd, sections)]
    )
    arrived = Screen(name=dest.lower(), title=dest, controls=[*tabs, *_maybe_destructive(rnd)])

    tap_args = {"kind": "tap", **target.selector()}
    assert_args = {"kind": "assert-visible", "arg": dest, "by": "text"}
    history: list[str] = []
    states = [
        State(goal, home, list(history), "next_step", tap_args, "direct_nav", package=package)
    ]
    history.append(_describe("next_step", tap_args))
    # Half of these confirm the destination and then check that the origin is behind them, so the
    # step after an `assert-visible` is another assertion as often as it is `done`.
    double_proof = rnd.random() < 0.5
    states.append(
        State(
            goal,
            arrived,
            list(history),
            "next_step",
            assert_args,
            "prove_arrival",
            package=package,
            known_screen=dest.lower(),
        )
    )
    history.append(_describe("next_step", assert_args))
    if double_proof:
        away = {"kind": "assert-not-visible", "arg": "Loading", "by": "text"}
        states.append(
            State(
                goal,
                arrived,
                list(history),
                "next_step",
                away,
                "prove_arrival",
                package=package,
                known_screen=dest.lower(),
            )
        )
        history.append(_describe("next_step", away))
    states.append(
        State(
            goal,
            arrived,
            list(history),
            "done",
            {},
            "finish",
            package=package,
            known_screen=dest.lower(),
        )
    )
    return states


def _states_multi_hop(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """The user's own example: two or three hops, then prove arrival.

    "open the app, navigate to the apps section, land on explore" is this family. The middle hop
    is the one a chooser cannot do, because the destination is not visible when the first decision
    is made.
    """

    pool = _SECTION_POOLS[split]
    hops = _pick(rnd, pool, rnd.randint(2, 3))
    dest = hops[-1]
    template = rnd.choice(_ROUTE_TEMPLATES[split][len(hops)])
    goal = template.format(**{f"h{index}": hop for index, hop in enumerate(hops)})
    package = _package(rnd, split)
    siblings = [s for s in pool if s not in hops]

    states: list[State] = []
    history: list[str] = []
    for depth, hop in enumerate(hops):
        # Each screen offers the next hop plus plausible siblings. The destination is only
        # nameable once its own screen is reached, so a single-screen chooser cannot solve this.
        others = _pick(rnd, tuple(siblings), rnd.randint(2, 4))
        here = [
            Control(section=hop, exposes=rnd.choice(_EXPOSURES)),
            *[Control(section=other, exposes=rnd.choice(_EXPOSURES)) for other in others],
        ]
        screen = Screen(
            name="home" if depth == 0 else hops[depth - 1].lower(),
            title=None if depth == 0 else hops[depth - 1],
            controls=[*here, *_maybe_destructive(rnd), *_noise(rnd, others)],
        )
        args = {"kind": "tap", **here[0].selector()}
        states.append(
            State(
                goal,
                screen,
                list(history),
                "next_step",
                args,
                "multi_hop",
                package=package,
                known_screen=None if depth == 0 else hops[depth - 1].lower(),
            )
        )
        history.append(_describe("next_step", args))
        # Prove the waypoint sometimes, then carry on. Without this, "the last history line is an
        # assert-visible" implies `done` at precision 1.000, and a trained model duly stopped early
        # whenever it saw one — the `premature_done` probe caught exactly that.
        if depth < len(hops) - 1 and rnd.random() < 0.70:
            waypoint = {"kind": "assert-visible", "arg": hop, "by": "text"}
            proved = Screen(
                name=hop.lower(),
                title=hop,
                controls=[*here, *_maybe_destructive(rnd)],
            )
            states.append(
                State(
                    goal,
                    proved,
                    list(history),
                    "next_step",
                    waypoint,
                    "multi_hop",
                    package=package,
                    known_screen=hop.lower(),
                )
            )
            history.append(_describe("next_step", waypoint))

    # A real destination screen still has its navigation and its own content, so `done` cannot be
    # inferred from "there is nothing left to click".
    arrived = Screen(
        name=dest.lower(),
        title=dest,
        controls=[
            Control(section=dest, exposes="text", clickable=False),
            *[
                Control(section=other, exposes=rnd.choice(_EXPOSURES))
                for other in _pick(rnd, tuple(x for x in pool if x != dest), rnd.randint(3, 5))
            ],
            *_maybe_destructive(rnd),
        ],
    )
    assert_args = {"kind": "assert-visible", "arg": dest, "by": "text"}
    states.append(
        State(
            goal,
            arrived,
            list(history),
            "next_step",
            assert_args,
            "prove_arrival",
            package=package,
            known_screen=dest.lower(),
        )
    )
    history.append(_describe("next_step", assert_args))
    states.append(
        State(
            goal,
            arrived,
            list(history),
            "done",
            {},
            "finish",
            package=package,
            known_screen=dest.lower(),
        )
    )
    return states


def _states_scroll_reveal(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """The destination exists but is below the fold: scroll-to first, then tap."""

    pool = _SECTION_POOLS[split]
    sections = _pick(rnd, pool, rnd.randint(4, 6))
    dest = sections[-1]
    goal = _goal_for(rnd, split, dest)
    package = _package(rnd, split)
    target = Control(section=dest, exposes=rnd.choice(("text", "text_and_rid")), offscreen=True)
    visible = [Control(section=s, exposes=rnd.choice(_EXPOSURES)) for s in sections[:-1]]
    listing = Screen(
        name="listing",
        title=None,
        controls=[
            *visible,
            target,
            Control(section="scroller", exposes="rid_only", clickable=False, scrollable=True),
            *_maybe_destructive(rnd),
        ],
    )
    revealed = Screen(
        name="listing",
        title=None,
        controls=[
            *visible,
            Control(section=dest, exposes=target.exposes),
            *_maybe_destructive(rnd),
        ],
    )
    arrived = Screen(
        name=dest.lower(),
        title=dest,
        controls=[
            *[
                Control(section=other, exposes=rnd.choice(_EXPOSURES))
                for other in _pick(rnd, tuple(x for x in pool if x != dest), rnd.randint(3, 5))
            ],
            *_maybe_destructive(rnd),
        ],
    )

    scroll_args = {"kind": "scroll-to", "arg": dest, "by": "text"}
    tap_args = {"kind": "tap", **Control(section=dest, exposes=target.exposes).selector()}
    assert_args = {"kind": "assert-visible", "arg": dest, "by": "text"}

    history: list[str] = []
    states = [
        State(
            goal, listing, list(history), "next_step", scroll_args, "scroll_reveal", package=package
        )
    ]
    history.append(_describe("next_step", scroll_args))
    states.append(
        State(
            goal, revealed, list(history), "next_step", tap_args, "scroll_reveal", package=package
        )
    )
    history.append(_describe("next_step", tap_args))
    states.append(
        State(
            goal,
            arrived,
            list(history),
            "next_step",
            assert_args,
            "prove_arrival",
            package=package,
            known_screen=dest.lower(),
        )
    )
    return states


def _states_already_there(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """Starts on the destination. The right answer is to prove and stop, never to act.

    The independent audit's sharpest V9 failure was the opposite of this: asked whether a row was
    on screen without changing anything, V9 chose to scroll 12/12. It had learned not to break
    things, not to keep its hands still.
    """

    pool = _SECTION_POOLS[split]
    dest = _pick(rnd, pool, 1)[0]
    goal = _goal_for(rnd, split, dest)
    package = _package(rnd, split)
    here = Screen(
        name=dest.lower(),
        title=dest,
        controls=[
            Control(section=dest, exposes="text", clickable=False),
            *[
                Control(section=s, exposes=rnd.choice(_EXPOSURES))
                for s in _pick(rnd, pool, rnd.randint(1, 3))
                if s != dest
            ],
            *_maybe_destructive(rnd),
        ],
    )
    assert_args = {"kind": "assert-visible", "arg": dest, "by": "text"}
    states = [
        State(
            goal,
            here,
            [],
            "next_step",
            assert_args,
            "already_there",
            package=package,
            known_screen=dest.lower(),
        )
    ]
    states.append(
        State(
            goal,
            here,
            [_describe("next_step", assert_args)],
            "done",
            {},
            "finish",
            package=package,
            known_screen=dest.lower(),
        )
    )
    return states


def _states_wrong_turn(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """History shows a hop that led somewhere unrelated: go back, then take the right one."""

    pool = _SECTION_POOLS[split]
    sections = _pick(rnd, pool, 4)
    dest, wrong = sections[0], sections[1]
    goal = _goal_for(rnd, split, dest)
    package = _package(rnd, split)
    # Density matches every other screen. When this was the only sparse screen in the corpus,
    # "few things to click" became sufficient for `key back` at 0.73 recall.
    stranded = Screen(
        name=wrong.lower(),
        title=wrong,
        controls=[
            Control(section=section, exposes=rnd.choice(_EXPOSURES))
            for section in _pick(rnd, tuple(x for x in pool if x != dest), rnd.randint(3, 6))
        ]
        + _maybe_destructive(rnd),
    )
    back_args = {"kind": "key", "arg": "back"}
    wrong_tap = {"kind": "tap", "label": wrong}
    home_target = Control(section=dest, exposes=rnd.choice(_EXPOSURES))
    home = Screen(
        name="home",
        title=None,
        controls=[
            home_target,
            *[Control(section=s, exposes=rnd.choice(_EXPOSURES)) for s in sections[1:]],
            *_maybe_destructive(rnd),
        ],
    )
    history = [_describe("next_step", wrong_tap)]
    states = [
        State(
            goal,
            stranded,
            list(history),
            "next_step",
            back_args,
            "wrong_turn_recovery",
            package=package,
            known_screen=wrong.lower(),
        )
    ]
    history.append("key back")
    states.append(
        State(
            goal,
            home,
            list(history),
            "next_step",
            {"kind": "tap", **home_target.selector()},
            "wrong_turn_recovery",
            package=package,
        )
    )
    return states


def _states_near_miss(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """Confusable neighbours. Only exact agreement with the goal counts."""

    pool = _SECTION_POOLS[split]
    dest = _pick(rnd, pool, 1)[0]
    goal = _goal_for(rnd, split, dest)
    package = _package(rnd, split)
    decoys = [_near_miss(rnd, dest) for _ in range(rnd.randint(1, 2))]
    target = Control(section=dest, exposes=rnd.choice(("text", "text_and_rid")))
    screen = Screen(
        name="home",
        title=None,
        controls=[
            target,
            *[Control(section=d, exposes="text") for d in decoys],
            *_maybe_destructive(rnd),
        ],
    )
    return [
        State(
            goal,
            screen,
            [],
            "next_step",
            {"kind": "tap", **target.selector()},
            "near_miss_label",
            package=package,
        )
    ]


def _states_breadcrumb(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """A passive heading reads the same as the clickable row. Pick the row."""

    pool = _SECTION_POOLS[split]
    dest = _pick(rnd, pool, 1)[0]
    goal = _goal_for(rnd, split, dest)
    package = _package(rnd, split)
    # The row carries a resource id; the heading is bare text with the same copy. Selecting by
    # label here would be ambiguous, so the resource id is the only unambiguous answer.
    row = Control(section=dest, exposes="rid_only")
    screen = Screen(
        name="section-list",
        title=dest,
        controls=[
            row,
            Control(section=dest, exposes="text", clickable=False, inert=True),
            *_maybe_destructive(rnd),
        ],
    )
    return [
        State(
            goal,
            screen,
            [],
            "next_step",
            {"kind": "tap", **row.selector()},
            "breadcrumb_vs_row",
            package=package,
        )
    ]


def _states_ime(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """A keyboard is up and it changes nothing: type into the field, or tap the visible target.

    This family exists to *break* the rule `ime -> hide-keyboard`, which the first draft taught at
    precision 1.000. Half of these states answer `input` and half answer `tap`, with the keyboard up
    in both, so the flag carries no information about the answer. `_decorate_states` then sprinkles
    the same flag across every other family too.
    """

    pool = _SECTION_POOLS[split]
    dest = _pick(rnd, pool, 1)[0]
    package = _package(rnd, split)
    if rnd.random() < 0.5:
        goal = rnd.choice(_SEARCH_TEMPLATES[split]).format(dest=dest)
        field = Control(section="searchField", exposes="rid_only")
        screen = Screen(name="search", title=None, controls=[field], ime_up=True)
        args: dict[str, Any] = {
            "kind": "input",
            "resource_id": field.rid,
            "text": dest,
        }
    else:
        goal = _goal_for(rnd, split, dest)
        target = Control(section=dest, exposes=rnd.choice(_EXPOSURES))
        others = [
            Control(section=other, exposes=rnd.choice(_EXPOSURES))
            for other in _pick(rnd, tuple(x for x in pool if x != dest), rnd.randint(3, 5))
        ]
        screen = Screen(name="home", title=None, controls=[target, *others], ime_up=True)
        args = {"kind": "tap", **target.selector()}
    return [State(goal, screen, [], "next_step", args, "keyboard_up_anyway", package=package)]


def _states_settling(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """Nothing on screen is actionable yet, so wait — and if waiting already failed twice, stop.

    The first draft made `settling -> wait-stable` a precision-1.000 rule. Here the deciding evidence
    is *content*: there is no clickable node to act on. And because "no clickable node" would then
    become the next shortcut, a third of these have already waited twice and must hand off instead.
    """

    pool = _SECTION_POOLS[split]
    dest = _pick(rnd, pool, 1)[0]
    goal = _goal_for(rnd, split, dest)
    package = _package(rnd, split)
    blank = Screen(
        name=None,
        title=None,
        controls=[Control(section="Loading", exposes="text", clickable=False)],
        loading=rnd.random() < 0.5,
    )
    if rnd.random() < 0.34:
        history = ["wait-stable", "wait-stable"]
        return [
            State(
                goal,
                blank,
                history,
                "handoff",
                {"reason": "no_progress"},
                "empty_screen_stalled",
                package=package,
            )
        ]
    target = Control(section=dest, exposes=rnd.choice(_EXPOSURES))
    ready = Screen(name="home", title=None, controls=[target])
    states = [
        State(
            goal, blank, [], "next_step", {"kind": "wait-stable"}, "empty_screen", package=package
        )
    ]
    states.append(
        State(
            goal,
            ready,
            ["wait-stable"],
            "next_step",
            {"kind": "tap", **target.selector()},
            "empty_screen",
            package=package,
        )
    )
    return states


def _states_input(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """Type a query, then do whatever the resulting screen actually calls for.

    Two defects fixed here. `history_tail_kind == "input" -> wait-for` was a precision-1.000,
    52.6x-lift rule in the first draft, because the step after typing was *always* a wait. Now the
    result is already on screen half the time, and then the right step is to tap it.

    And `submit` is gone. The helper's `case "input"` performs only `ACTION_SET_TEXT` — no IME
    action, no focus step, and it does not even check the return value. Teaching `submit=True`
    taught a flag the device silently ignores, which would have left every search unsubmitted and
    every following `wait-for` timing out.
    """

    pool = _SECTION_POOLS[split]
    dest = _pick(rnd, pool, 1)[0]
    goal = rnd.choice(_SEARCH_TEMPLATES[split]).format(dest=dest)
    package = _package(rnd, split)
    field = Control(section="searchField", exposes="rid_only")
    entry = Screen(name="search", title=None, controls=[field])
    input_args = {"kind": "input", "resource_id": field.rid, "text": dest}
    history: list[str] = []
    states = [State(goal, entry, [], "next_step", input_args, "input_then_read", package=package)]
    history.append(_describe("next_step", input_args))

    if rnd.random() < 0.5:
        # The result landed already: tap it. Nothing about the history says which of these it is.
        hit = Control(section=dest, exposes=rnd.choice(_EXPOSURES))
        shown = Screen(name="search-results", title=None, controls=[hit])
        states.append(
            State(
                goal,
                shown,
                list(history),
                "next_step",
                {"kind": "tap", **hit.selector()},
                "input_then_read",
                package=package,
            )
        )
    else:
        # Still empty of results, so wait for the one the goal named.
        pending = Screen(name="search-results", title=None, controls=[field])
        states.append(
            State(
                goal,
                pending,
                list(history),
                "next_step",
                {"kind": "wait-for", "arg": dest, "by": "text"},
                "input_then_read",
                package=package,
            )
        )
    return states


def _states_target_absent(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """A relevance refusal: the destination is nowhere in this app.

    This is the majority refusal shape, and V10 had it as the minority. Every control here is safe
    and authorized, so the only decidable rule is "nothing on this screen advances the goal" —
    which requires actually reading the goal.
    """

    pool = _SECTION_POOLS[split]
    present = _pick_leaving_one(rnd, pool, rnd.randint(3, 6))
    absent = next(s for s in pool if s not in present)
    goal = _goal_for(rnd, split, absent)
    package = _package(rnd, split)
    controls = [Control(section=s, exposes=rnd.choice(_EXPOSURES)) for s in present]
    screen = Screen(
        name="home",
        title=None,
        controls=[*controls, *_maybe_destructive(rnd), *_noise(rnd, present)],
    )
    return [
        State(
            goal,
            screen,
            [],
            "handoff",
            {"reason": "target_absent"},
            "target_absent_handoff",
            package=package,
        )
    ]


def _states_no_progress(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """A relevance refusal after real work: repeated hops have stopped changing anything."""

    pool = _SECTION_POOLS[split]
    present = _pick_leaving_one(rnd, pool, rnd.randint(3, 6))
    absent = next(s for s in pool if s not in present)
    goal = _goal_for(rnd, split, absent)
    package = _package(rnd, split)
    screen = Screen(
        name="home",
        title=None,
        controls=[Control(section=s, exposes=rnd.choice(_EXPOSURES)) for s in present]
        + _maybe_destructive(rnd),
    )
    # A stall is a stall because nothing advanced, not because two lines happen to be identical.
    # The first draft used one canned four-line loop for every case, which made
    # `history_repeats_a_line` a precision-1.000 predictor. These wander instead, and roughly half
    # never repeat a line at all.
    wander: list[str] = []
    for _ in range(rnd.randint(2, 4)):
        section = rnd.choice(present)
        verb = rnd.choice(("tap text=", "tap desc=", "tap rid="))
        wander.append(f"{verb}{section}")
        wander.append(rnd.choice(("key back", "key back", "wait-stable")))
    if rnd.random() < 0.5:
        wander = list(dict.fromkeys(wander))  # de-duplicate: a stall with no repeated line
    return [
        State(
            goal,
            screen,
            wander,
            "handoff",
            {"reason": "no_progress"},
            "no_progress_handoff",
            package=package,
        )
    ]


def _states_host_lane(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """The goal needs a capability an accessibility service does not have.

    An on-device driver cannot start a host proxy, copy a database through ``run-as``, install an
    APK or boot an emulator. Recognising its own lane is part of the job.
    """

    pool = _SECTION_POOLS[split]
    present = _pick(rnd, pool, rnd.randint(3, 6))
    capability = rnd.choice(HOST_ONLY_CAPABILITIES)
    # Phrasings are partitioned by split like every other goal string. A held-out host-lane goal
    # has to be recognised as host-lane from what it *asks for*, not from a sentence the model has
    # already seen tens of thousands of times.
    goal = rnd.choice(_HOST_LANE_GOALS[capability][split])
    # Compose rather than pick: a handful of fixed sentences per capability is memorisable, and
    # their shared surface statistics let capital-count and word-count answer the question outright.
    lead = rnd.choice(("", "I need you to ", "Could you ", "Next: ", "As a one-off, "))
    goal = lead + (goal[:1].lower() + goal[1:] if lead else goal)
    if rnd.random() < 0.35:
        # Terse forms, matching the length of a two-hop route goal. Without these, "two capitals in
        # a long sentence" was still sufficient for `needs_host_lane`.
        terse = {
            "proxy": "Proxy {a} and {b}",
            "database": "Dump {a} and {b} rows",
            "install": "Reinstall for {a} and {b}",
            "emulator": "Second target for {a} and {b}",
            "flags": "Flag {a} and {b} on",
            "logcat": "Logs for {a} and {b}",
            "clipboard_host": "Paste host text into {a} and {b}",
            "network_profile": "Throttle {a} and {b}",
        }[capability]
        first = rnd.choice(present)
        second = rnd.choice([x for x in present if x != first] or [first])
        goal = terse.format(a=first, b=second)
    # Every device goal in the first draft named a Capitalised Destination and no host-lane goal
    # did, so "are there any capitals?" answered the question outright. Naming a section that is
    # right there on screen makes the capability the only thing that decides it.
    if rnd.random() < 0.85:
        named = rnd.choice(present)
        second = rnd.choice([x for x in present if x != named] or [named])
        goal = rnd.choice(
            (
                f"{goal} for the {named} tab",
                f"On the {named} screen: {goal.lower()}",
                f"{named}: {goal.lower()}",
                f"{goal}, starting from {named}",
                f"{goal} across {named} and {second}",
                f"From {named}, then {second}: {goal.lower()}",
            )
        )
    # Length is spread independently of the capability so word count cannot separate the lanes.
    filler = rnd.choice(
        (
            "",
            " please",
            " when you get a chance",
            " and report back",
            " before anything else",
            " — this is the whole task",
        )
    )
    goal = goal + filler
    package = _package(rnd, split)
    screen = Screen(
        name="home",
        title=None,
        controls=[Control(section=s, exposes=rnd.choice(_EXPOSURES)) for s in present]
        + _maybe_destructive(rnd),
    )
    return [
        State(
            goal,
            screen,
            [],
            "handoff",
            {"reason": "needs_host_lane"},
            "host_lane_handoff",
            package=package,
        )
    ]


def _states_destructive_present(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """A destructive control is on screen and the goal is read-only. Navigate anyway.

    This family exists solely to break the V10 confound. Here "something destructive is visible"
    coincides with a *correct tap*, so the flag cannot be used as a refusal signal. Without states
    like these, refusal and destructiveness stay correlated no matter how the handoff families are
    weighted.
    """

    pool = _SECTION_POOLS[split]
    sections = _pick(rnd, pool, rnd.randint(2, 3))
    dest = sections[0]
    # Half of these open with the same destructive verb the authorization refusals use. Without
    # that, "the goal starts with Remove/Delete/Erase" predicts `needs_authorization` outright.
    goal = (
        rnd.choice(_AUTHORIZATION_TEMPLATES[split]).format(dest=dest)
        if rnd.random() < 0.5
        else _goal_for(rnd, split, dest)
    )
    package = _package(rnd, split)
    target = Control(section=dest, exposes=rnd.choice(_EXPOSURES))
    danger = [
        Control(section=label, exposes="text", destructive=True)
        for label in _pick(rnd, _DESTRUCTIVE_LABELS, rnd.randint(1, 3))
    ]
    screen = Screen(
        name="settings",
        title=None,
        controls=[
            target,
            *[Control(section=s, exposes=rnd.choice(_EXPOSURES)) for s in sections[1:]],
            *danger,
        ],
    )
    return [
        State(
            goal,
            screen,
            [],
            "next_step",
            {"kind": "tap", **target.selector()},
            "destructive_present_navigate_anyway",
            package=package,
        )
    ]


def _states_authorization(rnd: random.Random, split: str, ordinal: int) -> list[State]:
    """The minority refusal: the only way forward is a destructive control we may not press."""

    pool = _SECTION_POOLS[split]
    dest = _pick(rnd, pool, 1)[0]
    goal = rnd.choice(_AUTHORIZATION_TEMPLATES[split]).format(dest=dest)
    package = _package(rnd, split)
    danger = [
        Control(section=label, exposes="text", destructive=True)
        for label in _pick(rnd, _DESTRUCTIVE_LABELS, rnd.randint(1, 2))
    ]
    # Safe, authorized, and irrelevant company. Without it this screen would be the narrowest in
    # the corpus, and "few nodes" would predict refusal exactly the way candidate count predicted
    # V9's label. The semantics are unchanged: none of these advances a removal goal.
    bystanders = [
        Control(section=section, exposes=rnd.choice(_EXPOSURES))
        for section in _pick(rnd, tuple(s for s in pool if s != dest), rnd.randint(2, 5))
    ]
    screen = Screen(name="settings", title=None, controls=[*danger, *bystanders, *_noise(rnd, [])])
    return [
        State(
            goal,
            screen,
            [],
            "handoff",
            {"reason": "needs_authorization"},
            "authorization_handoff",
            package=package,
        )
    ]


#: Family generators and their relative weight. Acting families dominate, as they must — V9
#: regressed on taps precisely because a self-generated corpus dropped them to 40% of the signal.
#: Handoff families together are about a fifth, and within that, relevance refusals
#: (target_absent + no_progress + host_lane) outnumber authorization refusal roughly 4:1, which
#: is ``RELEVANCE_REFUSAL_SHARE`` inverted from V10's ratio.
_FAMILIES: tuple[tuple[str, Any, int], ...] = (
    ("direct_nav", _states_direct, 12),
    ("multi_hop", _states_multi_hop, 16),
    ("scroll_reveal", _states_scroll_reveal, 8),
    ("already_there", _states_already_there, 6),
    ("wrong_turn_recovery", _states_wrong_turn, 7),
    ("near_miss_label", _states_near_miss, 7),
    ("breadcrumb_vs_row", _states_breadcrumb, 6),
    ("keyboard_up_anyway", _states_ime, 6),
    ("empty_screen", _states_settling, 6),
    ("input_then_read", _states_input, 6),
    ("destructive_present_navigate_anyway", _states_destructive_present, 8),
    ("target_absent_handoff", _states_target_absent, 15),
    ("no_progress_handoff", _states_no_progress, 9),
    ("host_lane_handoff", _states_host_lane, 9),
    ("authorization_handoff", _states_authorization, 4),
)

FAMILY_NAMES = tuple(name for name, _, _ in _FAMILIES)


#: Rates at which screen *decoration* appears, applied to every state regardless of what the right
#: answer is. The first draft attached each flag to the one family whose answer it implied, so
#: `ime` predicted `hide-keyboard` and `settling` predicted `wait-stable`, both at precision 1.000
#: and lift above 55x. A trained model used exactly those rules. Real screens carry a scrollable
#: container almost always and a keyboard whenever a field has focus, entirely independently of what
#: the next step should be, so that is how they are generated here.
SCROLLABLE_RATE = 0.60
IME_RATE = 0.25
SETTLING_RATE = 0.20
#: How often a state's history gains a harmless detour, possibly a repeated one.
WANDER_RATE = 0.35
#: Detour destinations, shared across splits on purpose: a detour is not the answer to anything.
_WANDER_SECTIONS = ("Overview", "Recents", "Favourites", "Downloads", "Notifications")


def _decorate_states(states: list[State], rnd: random.Random) -> None:
    """Attach scrollable containers, keyboards and settling to states at fixed rates.

    Deliberately applied *after* the family has chosen its answer, and never consulted by it, so
    none of these three can carry information about the label.

    ``hide-keyboard`` is not taught anywhere in this corpus and this is why: the projection the
    device sends carries no bounds, so "the keyboard is covering the target" is not a fact the model
    can observe. Teaching it from the ``ime`` flag alone teaches a flag lookup and nothing else. The
    step still exists in the helper and the host can still use it; the on-device driver is simply not
    asked to guess at occlusion it cannot see.
    """

    for state in states:
        if rnd.random() < SCROLLABLE_RATE and not any(
            control.scrollable for control in state.screen.controls
        ):
            state.screen.controls.append(
                Control(section="scroller", exposes="rid_only", clickable=False, scrollable=True)
            )
        # A keyboard is up when something has focus, which is unrelated to the next decision.
        state.screen.ime_up = state.screen.ime_up or rnd.random() < IME_RATE
        # Prepend a plausible wander to the history, sometimes with a repeated line. Only stalls
        # revisited a screen in the first draft, so "did any line repeat?" answered
        # `handoff:no_progress` on its own. Wandering and then getting on with it is normal.
        if state.call != "handoff" and rnd.random() < WANDER_RATE:
            detour = rnd.choice(_WANDER_SECTIONS)
            prefix = [f"tap text={detour}", "key back"]
            if rnd.random() < 0.5:
                prefix += [f"tap text={detour}", "key back"]
            state.history[:0] = prefix
        # Only decorate `settling` where the screen still has something to act on; a genuinely
        # empty screen uses it as real evidence in `_states_empty_screen`.
        if any(control.clickable for control in state.screen.controls):
            state.screen.loading = state.screen.loading or rnd.random() < SETTLING_RATE


#: Node counts are drawn from ONE distribution shared by every family, then screens are padded up
#: to their draw. Left to itself, each family produces its own characteristic width — refusal
#: screens came out narrower than multi-hop screens — and width alone then predicts the label.
#: That is exactly the V9 defect (candidate count confounded with the answer) in a new costume.
#: Drawing the width first and padding to it makes node count carry no information at all.
_NODE_COUNT_CHOICES = (4, 5, 6, 7, 8, 9, 10)


def _pad_states(states: list[State], rnd: random.Random) -> None:
    """Pad every screen to an independently drawn node count with inert filler.

    Filler is always a non-clickable informational row drawn from ``_INERT_LABELS``. Those can
    never be a navigation destination and never match a section goal, so padding cannot change
    which answer is correct for any family — it only removes width as a signal.
    """

    for state in states:
        target = rnd.choice(_NODE_COUNT_CHOICES)
        # `as_context` counts visible controls plus the page title, so mirror that here.
        present = sum(1 for control in state.screen.controls if not control.offscreen)
        present += 1 if state.screen.title else 0
        for index in range(max(0, target - present)):
            state.screen.controls.append(
                Control(
                    section=f"{rnd.choice(_INERT_LABELS)} {index + 1}",
                    exposes="text",
                    clickable=False,
                    inert=True,
                )
            )


def group_id(split: str, family: str, ordinal: int) -> str:
    """A stable semantic-group identity, so no group can straddle a split boundary."""

    digest = hashlib.sha256(f"{SCHEMA}:{split}:{family}:{ordinal}".encode()).hexdigest()
    return f"v11-{family}-{digest[:12]}"


def generate(split: str, groups: int) -> Iterator[dict[str, Any]]:
    """Yield ``groups`` semantic groups for *split*, each a whole trajectory.

    A group is one journey. Expanding it into per-state rows happens in the curriculum module, so
    every state of one journey stays on the same side of a split boundary.
    """

    if split not in _SECTION_POOLS:
        raise ValueError(f"unknown split: {split!r}")
    weighted: list[tuple[str, Any]] = []
    for name, builder, weight in _FAMILIES:
        weighted.extend([(name, builder)] * weight)
    # Interleave deterministically. Built in declaration order, the weight table is a *sorted*
    # list, so ``ordinal % len(weighted)`` over fewer groups than its length would walk only the
    # leading families and silently drop the handoff ones off the end — a small run would contain
    # no refusals at all. Shuffling once with a fixed seed makes every prefix representative.
    random.Random(f"{SEED}:family-order").shuffle(weighted)

    for ordinal in range(groups):
        name, builder = weighted[ordinal % len(weighted)]
        # Seed per group, not per corpus: regenerating one family cannot shift another.
        rnd = random.Random(f"{SEED}:{split}:{name}:{ordinal}")
        states = builder(rnd, split, ordinal)
        # Width is drawn from a shared distribution, never inherited from the family.
        _pad_states(states, random.Random(f"{SEED}:width:{split}:{name}:{ordinal}"))
        # Screen decoration likewise: flags must not imply their own answer.
        _decorate_states(states, random.Random(f"{SEED}:decor:{split}:{name}:{ordinal}"))
        yield {
            "split": split,
            "family": name,
            "ordinal": ordinal,
            "group_id": group_id(split, name, ordinal),
            "states": states,
        }
