"""App exploration — mine an app's source tree for deeplinks and UI labels (PRD §6b).

Deeplinks are shortcuts: `aua open-and-analyze "myapp://catalog/books"` jumps straight to a
category instead of tapping through the app's menus. They're declared in the app's source —
AndroidManifest intent-filters and Compose/nav `navDeepLink`/`uriPattern` literals — so we
mine them once and save them to the app's playbook for the agent to reuse.

String resources (``res/values*/strings.xml``) are the app's on-screen labels in every
locale it ships. Mining them gives text lookups a cross-locale bridge: a query written in
one language resolves to what the device actually renders in its own locale.

Pure + offline: :func:`mine_deeplinks` / :func:`mine_strings` walk a directory and return
structured results; the engine saves them via the memory store. No device, no network.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Schemes that are (almost) never a useful in-app navigation deeplink — skip them when
# harvesting source literals so we don't record every https:// URL in the tree.
_NON_DEEPLINK_SCHEMES = frozenset(
    {"http", "https", "mailto", "tel", "sms", "smsto", "geo", "market", "content", "file", "data"}
)

_SOURCE_SUFFIXES = frozenset({".kt", ".kts", ".java", ".xml"})
_SKIP_DIRS = frozenset({".git", "build", "node_modules", ".gradle", ".idea", "dist", ".venv"})

# A quoted URI literal: scheme://rest. Scheme per RFC-3986 (letter then letter/digit/+-.).
_URI_RE = re.compile(r"""["']([a-z][a-z0-9+.\-]*://[^"'\s]+)["']""")
# Manifest intent-filter <data> attributes.
_SCHEME_RE = re.compile(r'android:scheme="([^"]+)"')
_HOST_RE = re.compile(r'android:host="([^"]+)"')


@dataclass
class MinedDeeplink:
    uri: str
    source: str  # repo-relative file where it was found
    templated: bool  # contains a placeholder ($x, {x}) or a trailing-slash stub


@dataclass
class MineResult:
    schemes: list[str] = field(default_factory=list)  # custom (non-web) schemes declared
    deeplinks: list[MinedDeeplink] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "schemes": self.schemes,
            "deeplinks": [
                {"uri": d.uri, "source": d.source, "templated": d.templated} for d in self.deeplinks
            ],
        }


def _is_templated(uri: str) -> bool:
    return bool(re.search(r"[$${}]", uri)) or uri.rstrip().endswith("/")


_TEST_DIR_PARTS = frozenset({"test", "androidTest", "testFixtures", "androidUnitTest"})


def _is_test_source(path: Path) -> bool:
    """Skip test sources — they hold throwaway example URIs (orders/123, task-456)."""
    if any(part in _TEST_DIR_PARTS for part in path.parts):
        return True
    return path.stem.endswith(("Test", "Tests", "Spec"))


def _iter_source_files(root: Path):
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in _SOURCE_SUFFIXES and not _is_test_source(path):
            yield path


def mine_deeplinks(root: Path, *, cap: int = 200) -> MineResult:
    """Scan *root* for custom-scheme deeplinks (manifest schemes + source literals).

    Custom schemes are discovered from AndroidManifest ``android:scheme`` declarations
    (minus web/mail schemes); source literals are then harvested for exactly those
    schemes so we record real navigation deeplinks, not every URL. Manifest
    scheme+host pairs (e.g. ``myapp`` + ``set-flags``) are reconstructed too.
    """
    root = root.expanduser()
    if not root.is_dir():
        return MineResult()
    schemes: set[str] = set()
    manifest_uris: set[str] = set()

    # Pass 1: manifests → custom schemes (+ scheme://host reconstruction).
    for mf in root.rglob("AndroidManifest.xml"):
        if any(part in _SKIP_DIRS for part in mf.parts):
            continue
        try:
            text = mf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = {s.lower() for s in _SCHEME_RE.findall(text)} - _NON_DEEPLINK_SCHEMES
        schemes |= found
        # crude scheme+host pairing within the file (host lines near a custom scheme)
        hosts = _HOST_RE.findall(text)
        for s in found:
            for h in hosts:
                manifest_uris.add(f"{s}://{h}")

    # Pass 2: source literals for the discovered custom schemes.
    found_uris: dict[str, str] = {}  # uri -> first source file
    for path in _iter_source_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "://" not in text:
            continue
        rel = _relpath(path, root)
        for uri in _URI_RE.findall(text):
            scheme = uri.split("://", 1)[0].lower()
            if scheme in _NON_DEEPLINK_SCHEMES:
                continue
            # A literal with an unknown scheme is only kept if the scheme was declared in
            # a manifest — keeps random "foo://bar" test strings out.
            if schemes and scheme not in schemes:
                continue
            found_uris.setdefault(uri, rel)

    deeplinks = [
        MinedDeeplink(uri=u, source=src, templated=_is_templated(u))
        for u, src in sorted(found_uris.items())
    ]
    # manifest scheme://host entries that didn't also appear as a literal
    for mu in sorted(manifest_uris):
        if mu not in found_uris:
            deeplinks.append(MinedDeeplink(uri=mu, source="AndroidManifest.xml", templated=False))
    deeplinks.sort(key=lambda d: d.uri)
    return MineResult(schemes=sorted(schemes), deeplinks=deeplinks[:cap])


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - defensive
        return path.name


# ------------------------------------------------------------------------ string mining

# values / values-es / values-es-rES. Non-locale qualifier dirs (values-night, values-v21,
# values-sw600dp, …) don't fit the 2–3-letter language shape and are skipped.
_VALUES_DIR_RE = re.compile(r"^values(?:-(?P<lang>[a-z]{2,3})(?:-r(?P<region>[A-Z]{2}))?)?$")
# BCP-47 dirs: values-b+sr+Latn → sr-Latn.
_BPLUS_DIR_RE = re.compile(r"^values-b\+(?P<tag>[a-zA-Z0-9+]+)$")
# 2–3-letter resource qualifiers that would otherwise parse as a language.
_NON_LOCALE_QUALIFIERS = frozenset({"car"})

DEFAULT_LOCALE = "default"


@dataclass
class MinedStrings:
    locales: list[str] = field(default_factory=list)
    entries: dict[str, dict[str, str]] = field(default_factory=dict)  # key → locale → text

    def as_dict(self) -> dict:
        return {"locales": self.locales, "entries": self.entries}


def _values_dir_locale(name: str) -> str | None:
    """Locale of a res ``values`` dir: ``values`` → default; ``values-es-rES`` → es-ES;
    ``values-b+sr+Latn`` → sr-Latn; qualifier-only dirs (``values-night``) → None."""
    m = _VALUES_DIR_RE.match(name)
    if m:
        lang = m.group("lang")
        if not lang:
            return DEFAULT_LOCALE
        if lang in _NON_LOCALE_QUALIFIERS:
            return None
        region = m.group("region")
        return f"{lang}-{region}" if region else lang
    m = _BPLUS_DIR_RE.match(name)
    if m:
        return m.group("tag").replace("+", "-")
    return None


def _clean_android_string(raw: str) -> str:
    """Undo Android resource escaping (surrounding quotes, ``\\'``, ``\\n``)."""
    s = raw.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n").strip()


def mine_strings(root: Path, *, cap: int = 5000) -> MinedStrings:
    """Harvest the app's per-locale UI labels from Android string resources.

    Keys are the stable cross-locale handle (``basket_edit``); values are what the
    device renders in each locale. The first definition of a key per locale wins
    (matching resource merge order closely enough for label lookup).
    """
    root = root.expanduser()
    out = MinedStrings()
    if not root.is_dir():
        return out
    locales: set[str] = set()
    for xml_path in sorted(root.rglob("strings.xml")):
        if any(part in _SKIP_DIRS for part in xml_path.parts) or _is_test_source(xml_path):
            continue
        locale = _values_dir_locale(xml_path.parent.name)
        if locale is None:
            continue
        try:
            tree = ET.parse(xml_path)
        except (ET.ParseError, OSError):
            continue
        for el in tree.getroot().iter("string"):
            key = el.get("name")
            if not key:
                continue
            if key not in out.entries and len(out.entries) >= cap:
                continue
            text = _clean_android_string("".join(el.itertext()))
            if not text:
                continue
            out.entries.setdefault(key, {}).setdefault(locale, text)
            locales.add(locale)
    out.locales = sorted(locales)
    return out
