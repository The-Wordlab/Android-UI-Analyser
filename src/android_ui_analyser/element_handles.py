"""Persistent, scoped element handles; semantic selectors remain independent of handles.

Records contain descriptor digests, not screen text or input values. Matching is one-to-one:
position never breaks a tie between otherwise indistinguishable items. Losing a record may
expire a handle, but must never make it address another element.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from .atomic import atomic_write_text
from .errors import DeviceError
from .leases import host_transaction
from .platforms.identity import TargetRef
from .platforms.options_transport import platform_options_fingerprint
from .schema import AppContext, Element, ElementId

if TYPE_CHECKING:
    from .engine import Engine

PREFIX = "el:"
MAX_RECORDS = 4096
_LOCAL_LIFETIMES: WeakKeyDictionary[Any, str] = WeakKeyDictionary()
# Only a trailing, explicit age on an independently labelled list item is state.
# Keep bare ages and arbitrary numbers verbatim: they may be the item's only identity.
_RELATIVE_AGE = re.compile(
    r"(?P<title>.*\S)\s+\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago$"
)


def is_handle(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _label(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _editable(element: Element) -> bool:
    return element.password or element.type.casefold() in {
        "edittext",
        "textfield",
        "textbox",
        "input",
        "textarea",
        "searchfield",
    }


def descriptors(elements: list[Element]) -> list[str]:
    """Identify a control by its role and semantic ancestry, including its owning row.

    Descendant labels distinguish repeated containers and their identical child buttons.
    Editable values and interaction flags are state. Coordinates and ordinal suffixes are
    deliberately absent from semantic descriptors, so scrolling/reordering cannot rename a row.
    """
    by_id = {el.id: el for el in elements}
    children: dict[ElementId | None, list[Element]] = {}
    for el in elements:
        children.setdefault(el.parent, []).append(el)
    sibling_roles = Counter((el.parent, el.resource_id, el.type) for el in elements)

    def identity_label(el: Element, value: str | None) -> str:
        label = _label(value)
        parent = by_id.get(el.parent) if el.parent is not None else None
        if (
            parent is not None and parent.scrollable
            and (el.clickable or el.long_clickable) and not _editable(el)
        ):
            age = _RELATIVE_AGE.fullmatch(label)
            if age is not None:
                # Keep an age marker, so a title-only row cannot inherit this identity.
                # This is descriptor-only; the original text is still published unchanged.
                return age["title"] + "\0relative-age"
        return label

    @cache
    def context_labels(parent: ElementId | None, window: str | None) -> str:
        # A dialog/page heading identifies what its identically named buttons operate on.
        # Restrict context to direct, passive labels so another row being inserted deeper
        # in a list does not rename the entire screen. System chrome has its own window.
        return _digest(
            sorted(
                {
                    label
                    for child in children.get(parent, [])
                    if child.window == window
                    and not children.get(child.id)
                    and not _editable(child)
                    and not (
                        child.clickable
                        or child.long_clickable
                        or child.checkable
                        or child.scrollable
                    )
                    if (label := _label(child.content_desc) or _label(child.text))
                }
            )
        )

    def subtree_labels(el: Element, visited: frozenset[ElementId] = frozenset()) -> list[str]:
        if el.id in visited:
            return []
        labels = [] if _editable(el) else [
            identity_label(el, el.content_desc), identity_label(el, el.text)
        ]
        for child in children.get(el.id, []):
            labels.extend(subtree_labels(child, visited | {el.id}))
        return sorted(set(filter(None, labels)))

    own: dict[ElementId, str] = {}
    for el in elements:
        # Full resource names retain package namespaces; a resource-id tail is a selector.
        label = identity_label(el, el.content_desc) or (
            "" if _editable(el) else identity_label(el, el.text)
        )
        anchor: Any = label
        if children.get(el.id) and not el.scrollable:
            anchor = subtree_labels(el)
        if el.scrollable:
            anchor = _label(el.content_desc)
        if not label and not el.resource_id and not children.get(el.id) and not _editable(el):
            # Opaque controls have only the visual/geometry evidence the perception layer
            # supplied. These conservative fallbacks can expire on rendering changes.
            from .identity import base_stable_key, stable_key

            anchor = base_stable_key(el.stable_key or stable_key(el))
        own[el.id] = _digest([el.window, el.resource_id, el.type, anchor])

    def ancestry(el: Element) -> list[Any]:
        # Only structural ownership supplies context. A new, unrelated top-level label
        # (or an OCR augmentation) must not rename every control already on the screen.
        parents: list[Any] = []
        seen = {el.id}
        parent = by_id.get(el.parent) if el.parent is not None else None
        while parent is not None and parent.id not in seen:
            seen.add(parent.id)
            # A layout's changing children are not its identity. Use descendant content only
            # for a repeated sibling container or an item under a scrollable parent.
            repeated = sibling_roles[(parent.parent, parent.resource_id, parent.type)] > 1
            grandparent = by_id.get(parent.parent) if parent.parent is not None else None
            row = repeated or bool(grandparent and grandparent.scrollable) or bool(parent.text)
            parents.append(
                own[parent.id]
                if row
                else [
                    parent.window,
                    parent.resource_id,
                    parent.type,
                    _label(parent.content_desc),
                    context_labels(parent.id, parent.window),
                ]
            )
            parent = grandparent
        return parents

    return [_digest([own[el.id], ancestry(el)]) for el in elements]


class HandleStore:
    def __init__(
        self,
        root: Path,
        target: TargetRef,
        scope: str,
        context: tuple[str | None, str | None] | None = None,
        context_reader: Callable[[], Any] | None = None,
    ) -> None:
        self.root = root
        self.path = root / "element-identities" / f"{target.storage_key}.json"
        self.lock_key = f"element-identities/{target.storage_key}"
        self.scope = scope
        self.context = context
        self.context_reader = context_reader

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                isinstance(data, dict)
                and data.get("version") == 1
                and data.get("scope") == self.scope
                and isinstance(data.get("records"), dict)
            ):
                handles = []
                for key, record in data["records"].items():
                    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                        break
                    if not isinstance(record, dict) or not isinstance(record.get("handle"), str):
                        break
                    handle = record["handle"]
                    if not re.fullmatch(r"el:[0-9a-f]{32}", handle):
                        break
                    handles.append(handle)
                else:
                    if len(set(handles)) == len(handles):
                        return data
        except (OSError, ValueError):
            pass
        # A corrupt/expired registry gets a new namespace, never recycled numeric handles.
        return {"version": 1, "scope": self.scope, "records": {}}

    def assign(
        self, elements: list[Element], *, app: str | None, surface: str | None
    ) -> list[Element]:
        if not elements:
            return []  # Empty launch/loading polls have no identities to scope or persist.
        if self.context_reader is not None:
            try:
                current = AppContext.coerce(self.context_reader())
                self.context = (current.app_id, current.surface_id)
            except Exception:
                self.context = None
        context = (
            [*self.context, app or self.context[0]] if self.context is not None else [app, surface]
        )
        keys = [_digest([context, key]) for key in descriptors(elements)]
        counts = Counter(keys)
        with host_transaction(self.root, self.lock_key):
            state = self._read()
            records = state["records"]
            result = []
            for el, key in zip(elements, keys, strict=True):
                prior = records.pop(key, None)
                handle = prior.get("handle") if isinstance(prior, dict) else None
                if counts[key] != 1 or not is_handle(handle):
                    handle = f"{PREFIX}{uuid4().hex}"
                # Ambiguous descriptors never get a reusable record. Each observation can
                # display those elements, but an action must refuse to guess their identity.
                if counts[key] == 1:
                    records[key] = {"handle": handle, "source": str(el.source.value)}
                result.append(el.model_copy(update={"handle": handle}))
            state["records"] = dict(list(records.items())[-MAX_RECORDS:])
            try:
                atomic_write_text(self.path, json.dumps(state, separators=(",", ":")))
            except OSError as exc:
                raise DeviceError("could not persist element identities") from exc
        return result

    def source(self, handle: str) -> str | None:
        with host_transaction(self.root, self.lock_key):
            for record in self._read()["records"].values():
                if isinstance(record, dict) and record.get("handle") == handle:
                    return str(record.get("source"))
        return None


def for_engine(engine: Engine, *, with_app: bool = True) -> HandleStore:
    device = engine.device
    root = Path(engine._lease_registry_dir).expanduser()
    target = TargetRef(engine.platform.name, device.serial)
    try:
        token = device.instance_token()
    except Exception:
        token = None
    # Without boot evidence an old process's identities cannot be safely adopted. The
    # connected Engine still has a useful local lifetime; reconnecting starts a new one.
    if not token:
        try:
            token = _LOCAL_LIFETIMES.setdefault(device, uuid4().hex)
        except TypeError:  # An unusual plugin runtime may not support weak references/hashing.
            token = engine._element_identity_lifetime
    scope = _digest(
        [
            token,
            platform_options_fingerprint(
                engine.config.platform_options(engine.platform.name), key_dir=root
            ),
        ]
    )
    # Capture context only once a nonempty tree needs identities. Repeated empty launch
    # polls must not turn a verified foreground into another native polling loop.
    return HandleStore(root, target, scope, context_reader=device.current_app if with_app else None)
