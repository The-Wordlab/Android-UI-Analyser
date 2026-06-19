"""Persistent per-app memory — the tool's long-term knowledge of an app (PRD §6b).

The tool builds and maintains a map of each app **itself** as it navigates: what screens
exist, what durable elements are on them, and the routes between them. An agent reads it
back at session start (``aua map``) so it knows the layout before navigating, and on every
``analyze`` the tool recognises the current screen (``meta.known_screen``) and records new
screens / route edges passively — no extra agent calls.

Design notes
------------
- **Durable skeleton only.** We store screens, routes, and *stable* elements (tabs,
  buttons, tool names, input *shapes*) — never volatile per-user content. A list of recent
  chats is recorded as a *shape* ("list (dynamic)"), not its items; the agent fetches live
  contents with ``analyze`` when it actually needs them. This keeps the map small, fresh,
  and free of PII.
- **Recognition by anchors.** A screen's identity is a set of durable *anchors* (stable
  resource-ids + short chrome labels + content-descriptions). Revisits are matched by
  Jaccard overlap (robust to dynamic content); large divergence or an app-version bump
  flags the screen ``stale`` so the agent re-verifies.
- **Privacy.** Local-only, never transmitted. ``EditText`` *values* are stored as a shape
  (``"<filled>"``), and secret / PII-looking text is redacted (``memory.redact``).
- **Single renderer.** :func:`render_map` produces both the on-disk ``MAP.md`` and every
  ``aua map`` view, so there is never any drift between them.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import MemoryCfg
    from .schema import Element

MEMORY_SCHEMA_VERSION = 1

# Jaccard overlap (after a small activity bonus) at/above which a screen is recognised as
# a known one rather than treated as new. Below the drift band it is a fresh screen.
_RECOGNIZE_MIN = 0.34

REDACT_TOKENS = {"<filled>", "<empty>", "<redacted>"}


# --------------------------------------------------------------------------- models


class KeyElement(BaseModel):
    """A durable, actionable element worth remembering (nav target, button, input)."""

    model_config = ConfigDict(extra="ignore")
    type: str
    label: str | None = None
    resource_id: str | None = None
    clickable: bool = False
    input: bool = False
    value: str | None = None  # shape only, never the literal value (e.g. "<filled>")


class ScreenRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    activity: str | None = None
    signature: str
    anchors: list[str] = Field(default_factory=list)
    tier: str = "hierarchy"
    key_elements: list[KeyElement] = Field(default_factory=list)
    dynamic: list[str] = Field(default_factory=list)  # shapes, e.g. "row list (dynamic)"
    notes: list[str] = Field(default_factory=list)
    app_version: str | None = None
    first_seen: str
    last_seen: str
    last_verified: str
    visit_count: int = 1
    stale: bool = False


class RouteEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")
    from_screen: str
    to_screen: str
    action: str  # human label, e.g. "tap 'Apps'"
    count: int = 1
    last_seen: str


class AppMap(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: int = MEMORY_SCHEMA_VERSION
    package: str
    label: str | None = None
    app_version: str | None = None
    last_verified: str | None = None
    screens: dict[str, ScreenRecord] = Field(default_factory=dict)
    routes: list[RouteEdge] = Field(default_factory=list)


class SessionState(BaseModel):
    """Cross-process navigation cursor (per device serial) used to draw route edges."""

    model_config = ConfigDict(extra="ignore")
    package: str | None = None
    current_screen: str | None = None
    pending: list[str] = Field(default_factory=list)  # action summaries since last analyze


class RecordOutcome(NamedTuple):
    name: str
    was_known: bool
    stale: bool
    created: bool


# --------------------------------------------------------------------------- redaction


def _is_input(el: Element) -> bool:
    t = (el.type or "").lower()
    return any(k in t for k in ("edittext", "textfield", "autocomplete", "searchview"))


_SECRET_HINT = re.compile(
    r"pass(word|code)|secret|otp|cvv|\bpin\b|token|credit.?card|card.?number|security.?code",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)\+?\d[\d ().-]{7,}\d(?!\d)")
_LONGNUM = re.compile(r"\d{6,}")


def _is_secret_field(el: Element) -> bool:
    parts = " ".join(p for p in (el.resource_id, el.content_desc, el.type) if p)
    return bool(_SECRET_HINT.search(parts))


def _looks_pii(text: str) -> bool:
    return bool(_EMAIL.search(text) or _PHONE.search(text) or _LONGNUM.search(text))


def _looks_dynamic(text: str) -> bool:
    """Heuristic: volatile content (counts, prices, clocks, ids) we must not anchor on."""
    t = text.strip()
    if not t:
        return True
    digits = sum(c.isdigit() for c in t)
    if digits and digits / len(t) > 0.4:
        return True
    return bool(re.search(r"\d{1,2}:\d{2}", t))


def _id_tail(resource_id: str | None) -> str | None:
    if not resource_id:
        return None
    tail = resource_id.split("/")[-1].strip()
    return tail or None


# System chrome (status bar, nav bar) is shared across every screen, so it is a poor
# identity signal — exclude it from signatures/names. We keep it out of the *signature*
# only; ``analyze`` output is unchanged.
_SYSTEM_ID_PREFIXES = ("com.android.systemui:", "android:")


def _system_chrome(el: Element, height: int | None) -> bool:
    """True for status-bar / framework chrome (never an app's own content)."""
    rid = el.resource_id or ""
    if rid.startswith(_SYSTEM_ID_PREFIXES):
        return True
    # Status-bar band (battery/wifi/clock icons often carry only a content-desc).
    return height is not None and el.center[1] < 0.035 * height


def _bottom_nav(el: Element, height: int | None) -> bool:
    """True for the persistent bottom navigation band (shared across top-level tabs)."""
    return height is not None and el.center[1] > 0.90 * height


def redact_label(el: Element, *, redact: bool = True) -> str | None:
    """Durable, privacy-safe label for an element (never a typed value)."""
    if _is_input(el):
        hint = el.content_desc or (_id_tail(el.resource_id) or "").replace("_", " ").strip() or None
        if redact and hint and _is_secret_field(el):
            return "<redacted>"
        return hint  # the field VALUE (el.text) is deliberately never read here
    text = (el.text or el.content_desc or "").strip()
    if not text:
        return None
    if redact and (_is_secret_field(el) or _looks_pii(text)):
        return "<redacted>"
    return text[:60]


# --------------------------------------------------------- signatures & key elements


def screen_anchors(
    elements: list[Element], *, redact: bool = True, height: int | None = None
) -> set[str]:
    """Durable identity tokens for a screen (stable ids + chrome labels + descriptions).

    System chrome (status bar) and the persistent bottom-nav band are excluded so the
    signature is driven by the screen *body* — otherwise every tab looks alike.
    """
    anchors: set[str] = set()
    for el in elements:
        if _system_chrome(el, height) or _bottom_nav(el, height):
            continue
        tail = _id_tail(el.resource_id)
        if tail and not _looks_dynamic(tail):
            anchors.add("id:" + tail.lower())
        cd = (el.content_desc or "").strip()
        if (
            cd
            and len(cd) <= 30
            and not _looks_dynamic(cd)
            and not (redact and (_is_secret_field(el) or _looks_pii(cd)))
        ):
            anchors.add("cd:" + cd.lower())
        if el.text and not _is_input(el):
            t = el.text.strip()
            if (
                1 <= len(t) <= 24
                and not _looks_dynamic(t)
                and not (redact and (_is_secret_field(el) or _looks_pii(t)))
            ):
                anchors.add("tx:" + t.lower())
    return anchors


def signature(activity: str | None, anchors: set[str]) -> str:
    base = (activity or "") + "|" + "\n".join(sorted(anchors))
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def key_elements(
    elements: list[Element], *, redact: bool = True, cap: int = 40, height: int | None = None
) -> list[KeyElement]:
    """The durable, actionable subset of a screen (nav, buttons, inputs) — deduped.

    Status-bar chrome is dropped; the bottom nav is kept (it is useful app navigation).
    """
    out: list[KeyElement] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for el in elements:
        if _system_chrome(el, height):
            continue
        is_in = _is_input(el)
        if not (el.clickable or is_in or el.resource_id):
            continue
        label = redact_label(el, redact=redact)
        tail = _id_tail(el.resource_id)
        key = (el.type, label, tail)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            KeyElement(
                type=el.type,
                label=label,
                resource_id=tail,
                clickable=el.clickable,
                input=is_in,
                value=("<filled>" if (is_in and (el.text or "").strip()) else None),
            )
        )
        if len(out) >= cap:
            break
    return out


def detect_dynamic(elements: list[Element]) -> list[str]:
    """Recognise repeated-row shapes (lists) and record them as shapes, not contents."""
    counts: Counter[str] = Counter()
    for el in elements:
        tail = _id_tail(el.resource_id)
        if tail and not _looks_dynamic(tail):
            counts[tail] += 1
    shapes = [f"{tail} list (dynamic, {n}+ items)" for tail, n in counts.items() if n >= 4]
    return shapes[:5]


def title_of(elements: list[Element], height: int | None = None) -> str | None:
    """Topmost short, non-dynamic heading text (below the status bar) — a name heuristic."""
    if not elements:
        return None
    h = height or max((e.bounds[3] for e in elements), default=1) or 1
    cands: list[tuple[int, int, str]] = []
    for el in elements:
        if _is_input(el) or _system_chrome(el, height):
            continue
        t = (el.text or el.content_desc or "").strip()
        if not t or _looks_dynamic(t) or not (2 <= len(t) <= 24):
            continue
        if el.center[1] <= 0.22 * h:
            cands.append((el.center[1], -len(t), t))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def slug(text: str | None) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return s.strip("_")[:40]


def _short(text: str | None, tokens: int = 2) -> str:
    """A short slug: first ``tokens`` words only (so a long card label → a clean name)."""
    parts = [p for p in slug(text).split("_") if p]
    return "_".join(parts[:tokens])


def propose_name(
    *,
    hint: str | None = None,
    inbound_label: str | None = None,
    inbound_kind: str | None = None,
    title: str | None = None,
    activity: str | None = None,
    is_first: bool = False,
) -> str:
    cands: list[str] = []
    if hint:
        cands.append(slug(hint))  # an explicit name is used verbatim
    if inbound_kind == "tap" and inbound_label:
        cands.append(_short(inbound_label))  # nav taps give clean names ("Apps" → apps)
    if title:
        cands.append(_short(title))
    if inbound_label:
        cands.append(_short(inbound_label))
    if is_first:
        cands.append("home")
    if activity:
        cands.append(_short(activity.rsplit(".", 1)[-1].replace("Activity", "")))
    for c in cands:
        if c:
            return c
    return "screen"


# --------------------------------------------------------------------------- store


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class AppMemoryStore:
    """Read/write the per-app maps and the per-device navigation session."""

    def __init__(self, cfg: MemoryCfg) -> None:
        self.cfg = cfg

    # -- paths (everything stays under memory.dir) ------------------------

    @property
    def base(self) -> Path:
        return Path(self.cfg.dir).expanduser()

    def memory_root(self) -> Path:
        return self.base / "memory"

    def app_dir(self, package: str) -> Path:
        return self.memory_root() / _safe(package)

    def index_path(self, package: str) -> Path:
        return self.app_dir(package) / "index.json"

    def map_path(self, package: str) -> Path:
        return self.app_dir(package) / "MAP.md"

    def session_path(self, serial: str) -> Path:
        return self.base / "state" / f"session_{_safe(serial)}.json"

    # -- app map I/O ------------------------------------------------------

    def load(self, package: str) -> AppMap | None:
        path = self.index_path(package)
        if not path.is_file():
            return None
        try:
            return AppMap.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - corrupt file → treat as absent
            return None

    def save(self, app: AppMap) -> None:
        d = self.app_dir(app.package)
        d.mkdir(parents=True, exist_ok=True)
        self.index_path(app.package).write_text(app.model_dump_json(indent=2), encoding="utf-8")
        self.map_path(app.package).write_text(render_map(app, detail="default"), encoding="utf-8")

    def list_apps(self) -> list[str]:
        root = self.memory_root()
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if (p / "index.json").is_file())

    # -- session I/O ------------------------------------------------------

    def load_session(self, serial: str) -> SessionState:
        path = self.session_path(serial)
        if path.is_file():
            try:
                return SessionState.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover
                pass
        return SessionState()

    def save_session(self, serial: str, sess: SessionState) -> None:
        path = self.session_path(serial)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sess.model_dump_json(), encoding="utf-8")

    # -- recording --------------------------------------------------------

    def _recognize(
        self, app: AppMap, anchors: set[str], activity: str | None, sig: str
    ) -> tuple[str | None, float]:
        best: str | None = None
        best_sim = 0.0
        for name, rec in app.screens.items():
            if rec.signature == sig:
                return name, 1.0
            sim = jaccard(anchors, set(rec.anchors))
            if rec.activity and activity:
                sim += 0.05 if rec.activity == activity else -0.10
            if sim > best_sim:
                best, best_sim = name, sim
        if best is not None and best_sim >= _RECOGNIZE_MIN:
            return best, best_sim
        return None, best_sim

    def _unique_name(self, app: AppMap, base: str) -> str:
        base = base or "screen"
        if base not in app.screens:
            return base
        i = 2
        while f"{base}_{i}" in app.screens:
            i += 1
        return f"{base}_{i}"

    def _rename(self, app: AppMap, old: str, new: str) -> str:
        if new == old or not new:
            return old
        new = self._unique_name(app, new)
        rec = app.screens.pop(old)
        rec.name = new
        app.screens[new] = rec
        for e in app.routes:
            if e.from_screen == old:
                e.from_screen = new
            if e.to_screen == old:
                e.to_screen = new
        return new

    def record_screen(
        self,
        *,
        package: str,
        elements: list[Element],
        label: str | None = None,
        activity: str | None = None,
        app_version: str | None = None,
        tier: str = "hierarchy",
        name_hint: str | None = None,
        inbound_label: str | None = None,
        inbound_kind: str | None = None,
        screen_height: int | None = None,
    ) -> RecordOutcome:
        """Record/update the current screen; return how it was identified."""
        if not self.cfg.enabled:
            return RecordOutcome(name="", was_known=False, stale=False, created=False)
        now = _now_iso()
        app = self.load(package) or AppMap(package=package)
        anchors = screen_anchors(elements, redact=self.cfg.redact, height=screen_height)
        sig = signature(activity, anchors)
        name, _sim = self._recognize(app, anchors, activity, sig)
        was_known = name is not None
        created = False
        stale = False

        if name is None:
            created = True
            base = name_hint or propose_name(
                inbound_label=inbound_label,
                inbound_kind=inbound_kind,
                title=title_of(elements, screen_height),
                activity=activity,
                is_first=len(app.screens) == 0,
            )
            name = self._unique_name(app, slug(base) or "screen")
            app.screens[name] = ScreenRecord(
                name=name,
                activity=activity,
                signature=sig,
                anchors=sorted(anchors),
                tier=tier,
                key_elements=key_elements(elements, redact=self.cfg.redact, height=screen_height),
                dynamic=detect_dynamic(elements),
                app_version=app_version,
                first_seen=now,
                last_seen=now,
                last_verified=now,
            )
        else:
            if name_hint and slug(name_hint) and slug(name_hint) != name:
                name = self._rename(app, name, slug(name_hint))
            rec = app.screens[name]
            if app_version and rec.app_version and app_version != rec.app_version:
                stale = True
            if (1.0 - jaccard(anchors, set(rec.anchors))) > self.cfg.drift_threshold:
                stale = True
            rec.last_seen = now
            rec.visit_count += 1
            if app_version:
                rec.app_version = app_version
            rec.stale = stale
            if not stale:
                rec.last_verified = now
                rec.signature = sig
                rec.anchors = sorted(anchors)
                rec.tier = tier
                if ke := key_elements(elements, redact=self.cfg.redact, height=screen_height):
                    rec.key_elements = ke
                if dyn := detect_dynamic(elements):
                    rec.dynamic = dyn

        if app_version:
            app.app_version = app_version
        if label:
            app.label = label
        app.last_verified = now
        self.save(app)
        return RecordOutcome(name=name, was_known=was_known, stale=stale, created=created)

    def record_route(self, package: str, from_screen: str, to_screen: str, action: str) -> None:
        if not self.cfg.enabled or from_screen == to_screen:
            return
        app = self.load(package)
        if app is None:
            return
        now = _now_iso()
        for e in app.routes:
            if e.from_screen == from_screen and e.to_screen == to_screen and e.action == action:
                e.count += 1
                e.last_seen = now
                self.save(app)
                return
        app.routes.append(
            RouteEdge(from_screen=from_screen, to_screen=to_screen, action=action, last_seen=now)
        )
        self.save(app)

    # -- auto-record orchestration (engine + daemon call these) -----------

    def observe_action(self, serial: str, summary: str) -> None:
        """Remember the last state-changing action so the next analyze can draw an edge."""
        if not (self.cfg.enabled and self.cfg.auto_record):
            return
        sess = self.load_session(serial)
        sess.pending.append(summary)
        sess.pending = sess.pending[-6:]
        self.save_session(serial, sess)

    def observe_screen(
        self,
        serial: str,
        *,
        package: str,
        elements: list[Element],
        label: str | None = None,
        activity: str | None = None,
        app_version: str | None = None,
        tier: str = "hierarchy",
        screen_height: int | None = None,
    ) -> str | None:
        """Record the current screen + any pending route edge. Returns ``known_screen``."""
        if not (self.cfg.enabled and self.cfg.auto_record):
            return None
        sess = self.load_session(serial)
        prev, prev_pkg, pending = sess.current_screen, sess.package, list(sess.pending)
        inbound_label, inbound_kind = _parse_inbound(pending)
        outcome = self.record_screen(
            package=package,
            elements=elements,
            label=label,
            activity=activity,
            app_version=app_version,
            tier=tier,
            inbound_label=inbound_label,
            inbound_kind=inbound_kind,
            screen_height=screen_height,
        )
        if pending and prev and prev_pkg == package and prev != outcome.name:
            action = " + ".join(pending) if len(pending) <= 3 else f"{pending[0]} … {pending[-1]}"
            self.record_route(package, prev, outcome.name, action)
        sess.current_screen = outcome.name
        sess.package = package
        sess.pending = []
        self.save_session(serial, sess)
        return outcome.name if outcome.was_known else None

    # -- management -------------------------------------------------------

    def forget(self, package: str, screen: str | None = None) -> dict[str, str | None]:
        if screen is None:
            d = self.app_dir(package)
            if d.is_dir():
                shutil.rmtree(d)
                return {"forgot": package}
            return {"forgot": None}
        app = self.load(package)
        if app and screen in app.screens:
            del app.screens[screen]
            app.routes = [
                e for e in app.routes if e.from_screen != screen and e.to_screen != screen
            ]
            self.save(app)
            return {"forgot": f"{package}/{screen}"}
        return {"forgot": None}


def _parse_inbound(pending: list[str]) -> tuple[str | None, str | None]:
    """Pull a (label, kind) from action summaries for naming a destination screen."""
    label: str | None = None
    kind: str | None = None
    for s in pending:
        m = re.match(r"([\w-]+)\s+'(.+?)'", s)
        if m:
            kind, lbl = m.group(1), m.group(2)
            label = None if lbl in REDACT_TOKENS else lbl
    return label, kind


# --------------------------------------------------------------------------- rendering


def _adjacency(app: AppMap) -> dict[str, list[RouteEdge]]:
    adj: dict[str, list[RouteEdge]] = {}
    for e in app.routes:
        adj.setdefault(e.from_screen, []).append(e)
    return adj


def _roots(app: AppMap) -> list[str]:
    targets = {e.to_screen for e in app.routes}
    roots = [n for n in app.screens if n not in targets]
    if not roots:
        roots = ["home"] if "home" in app.screens else list(app.screens)[:1]
    roots.sort(key=lambda n: (0 if n == "home" else 1, app.screens[n].first_seen))
    return roots


def _summarize_keys(rec: ScreenRecord) -> list[str]:
    actions: list[str] = []
    for ke in rec.key_elements:
        if ke.clickable and not ke.input and ke.label and ke.label not in actions:
            actions.append(ke.label)
    out: list[str] = []
    if actions:
        shown = " | ".join(actions[:10]) + (" …" if len(actions) > 10 else "")
        out.append(f"actions: {shown}")
    for ke in [k for k in rec.key_elements if k.input][:3]:
        val = f" ({ke.value})" if ke.value else ""
        out.append(f"input: {ke.label or 'field'}{val}")
    return out


def _render_tree(
    app: AppMap,
    name: str,
    adj: dict[str, list[RouteEdge]],
    seen: set[str],
    lines: list[str],
    *,
    prefix: str,
    detail: str,
    depth: int | None,
    inbound: str | None = None,
    level: int = 0,
) -> None:
    rec = app.screens.get(name)
    via = f"  [{inbound}]" if inbound else ""
    if name in seen:
        lines.append(f"{prefix}{name}{via} ↩")
        return
    seen.add(name)
    tier = f"  (tier: {rec.tier})" if rec else ""
    stale = "  [STALE]" if rec and rec.stale else ""
    lines.append(f"{prefix}{name}{tier}{stale}{via}")
    kid = prefix + "    "
    if detail != "brief" and rec:
        lines.extend(f"{kid}- {s}" for s in _summarize_keys(rec))
        lines.extend(f"{kid}- {d}" for d in rec.dynamic)
    if depth is not None and level >= depth:
        return
    for e in sorted(adj.get(name, []), key=lambda x: x.to_screen):
        _render_tree(
            app,
            e.to_screen,
            adj,
            seen,
            lines,
            prefix=kid,
            detail=detail,
            depth=depth,
            inbound=e.action,
            level=level + 1,
        )


def _header(app: AppMap) -> list[str]:
    lines = [f"# {app.label or app.package}  ({app.package})"]
    meta: list[str] = []
    if app.app_version:
        meta.append(f"version {app.app_version}")
    if app.last_verified:
        meta.append(f"last verified {app.last_verified}")
    n, r = len(app.screens), len(app.routes)
    meta.append(f"{n} screen{'s' * (n != 1)}, {r} route{'s' * (r != 1)}")
    lines.append("_" + " · ".join(meta) + "_")
    return lines


def render_map(
    app: AppMap,
    *,
    detail: str = "default",
    find: str | None = None,
    screen: str | None = None,
    depth: int | None = None,
) -> str:
    """The single source for ``MAP.md`` and every ``aua map`` view (PRD §6b)."""
    if find:
        return _render_find(app, find)
    if screen:
        return _render_screen_detail(app, screen)

    lines = [*_header(app), ""]
    if not app.screens:
        lines.append("_(no screens recorded yet — run `aua analyze` while navigating)_")
        return "\n".join(lines) + "\n"

    lines.append("## Screens")
    adj = _adjacency(app)
    seen: set[str] = set()
    for root in _roots(app):
        _render_tree(app, root, adj, seen, lines, prefix="", detail=detail, depth=depth)
    for name in app.screens:  # any screens not reachable from a root
        if name not in seen:
            _render_tree(app, name, adj, seen, lines, prefix="", detail=detail, depth=depth)

    if app.routes:
        lines.append("")
        lines.append("## Routes")
        for e in sorted(app.routes, key=lambda x: (x.from_screen, x.to_screen, x.action)):
            lines.append(f"{e.from_screen} --{e.action}--> {e.to_screen}")
    return "\n".join(lines).rstrip() + "\n"


def _shortest_path(app: AppMap, target: str) -> list[RouteEdge]:
    """Shortest route from a root (else any screen) to *target*; [] if it is a root."""
    adj = _adjacency(app)
    roots = _roots(app)
    if target in roots:
        return []
    starts = roots + [n for n in app.screens if n not in roots]
    for start in starts:
        visited = {start}
        queue: deque[tuple[str, list[RouteEdge]]] = deque([(start, [])])
        while queue:
            node, path = queue.popleft()
            for e in adj.get(node, []):
                if e.to_screen in visited:
                    continue
                new_path = path + [e]
                if e.to_screen == target:
                    return new_path
                visited.add(e.to_screen)
                queue.append((e.to_screen, new_path))
    return []


def _format_path(path: list[RouteEdge]) -> str:
    parts = [path[0].from_screen]
    parts += [f"--{e.action}--> {e.to_screen}" for e in path]
    return " ".join(parts)


def _find_targets(app: AppMap, query: str) -> list[str]:
    """Screens matching *query* by name, key-element label, anchor, dynamic shape, or a
    route action that leads to them (so ``--find "image"`` finds the image screen)."""
    q = query.lower().strip()

    def matches(name: str) -> bool:
        rec = app.screens[name]
        if q in name.lower():
            return True
        if any(ke.label and q in ke.label.lower() for ke in rec.key_elements):
            return True
        if any(q in a.lower() for a in rec.anchors):
            return True
        return any(q in d.lower() for d in rec.dynamic)

    targets = [n for n in app.screens if matches(n)]
    for e in app.routes:
        if q in e.action.lower() and e.to_screen not in targets:
            targets.append(e.to_screen)
    return targets


def find_result(app: AppMap, query: str) -> dict[str, object]:
    """Structured ``--find --json`` payload: matching screens + the route to each."""
    results = []
    for t in _find_targets(app, query):
        rec = app.screens.get(t)
        path = _shortest_path(app, t)
        results.append(
            {
                "screen": t,
                "tier": rec.tier if rec else None,
                "stale": rec.stale if rec else None,
                "route": [
                    {"from": e.from_screen, "action": e.action, "to": e.to_screen} for e in path
                ],
                "key_elements": [ke.model_dump() for ke in rec.key_elements] if rec else [],
            }
        )
    return {"query": query, "package": app.package, "results": results}


def _render_find(app: AppMap, query: str) -> str:
    targets = _find_targets(app, query)
    lines = [f"# find: {query}  ({app.package})"]
    if not targets:
        lines.append("")
        lines.append("_(no matching screen in memory — navigate there once so it is recorded)_")
        return "\n".join(lines) + "\n"
    for t in targets:
        rec = app.screens.get(t)
        lines.append("")
        lines.append(f"## {t}" + (f"  (tier: {rec.tier})" if rec else ""))
        path = _shortest_path(app, t)
        lines.append("route: " + (_format_path(path) if path else "(start here)"))
        if rec:
            lines.extend(f"  - {s}" for s in _summarize_keys(rec))
    return "\n".join(lines).rstrip() + "\n"


def _render_screen_detail(app: AppMap, screen: str) -> str:
    rec = app.screens.get(screen)
    if rec is None:
        avail = ", ".join(sorted(app.screens)) or "(none)"
        return f"# {screen}\n\n_(unknown screen; known: {avail})_\n"
    lines = [f"# {screen}  ({app.package})"]
    meta = [f"tier: {rec.tier}", f"visits: {rec.visit_count}", f"last verified {rec.last_verified}"]
    if rec.activity:
        meta.insert(0, f"activity: {rec.activity}")
    if rec.stale:
        meta.append("STALE")
    lines.append("_" + " · ".join(meta) + "_")
    if rec.key_elements:
        lines.append("")
        lines.append("## Elements")
        for ke in rec.key_elements:
            bits = [ke.type]
            if ke.label:
                bits.append(f"“{ke.label}”")
            if ke.resource_id:
                bits.append(f"#{ke.resource_id}")
            flags = [f for f, on in (("clickable", ke.clickable), ("input", ke.input)) if on]
            if ke.value:
                flags.append(ke.value)
            if flags:
                bits.append(f"[{', '.join(flags)}]")
            lines.append("- " + " ".join(bits))
    if rec.dynamic:
        lines.append("")
        lines.append("## Dynamic")
        lines.extend(f"- {d}" for d in rec.dynamic)
    incoming = [e for e in app.routes if e.to_screen == screen]
    outgoing = [e for e in app.routes if e.from_screen == screen]
    if incoming or outgoing:
        lines.append("")
        lines.append("## Routes")
        for e in incoming:
            lines.append(f"← {e.from_screen} --{e.action}-->")
        for e in outgoing:
            lines.append(f"→ {e.to_screen} ({e.action})")
    return "\n".join(lines).rstrip() + "\n"
