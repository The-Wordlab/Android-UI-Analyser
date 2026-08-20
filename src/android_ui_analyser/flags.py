"""Feature-flag helper: apply KEY=VAL via a package-specific deeplink, then read it back.

Templates come from user config (``flags.templates``) only — a set-flags scheme is an
app's private contract, and firing one that no installed build declares is
indistinguishable from success. The read-back is what turns "a deeplink was fired" into
"these keys are set": on a debuggable build the overrides are readable straight out of the
app's own ``shared_prefs`` XML.

:func:`write_prefs` is the other direction, for the state no deeplink and no screen exposes
(which backend a build talks to, whether onboarding is already "seen"). It is the same
``run-as`` surface and the same XML dialect, written instead of read — plus a host-side
restore point, because a preference AUA writes outlives the agent that wrote it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from shlex import quote
from typing import Any, NamedTuple, Protocol
from urllib.parse import quote as urlquote
from urllib.parse import urlencode
from xml.etree import ElementTree

import yaml

from .atomic import atomic_write_text
from .errors import DeviceError, UsageError

# One prefs entry, attribute-valued (`<boolean … value="true" />`) or text-valued
# (`<string …>a</string>`); the backreference stops the two shapes from crossing over.
_ENTRY_RE = re.compile(
    r'<(string|boolean|int|long|float)\s+name="(?P<key>[^"]+)"'
    r'(?:\s+value="(?P<attr>[^"]*)"\s*/>|\s*>(?P<text>[^<]*)</\1>)'
)
_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
# A prefs file is addressed by basename inside the app's own `shared_prefs`, never by path:
# a write is a replacement, and a traversal would replace something else entirely.
_PREFS_FILE_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.xml$")
# Exactly the declaration Android's own SharedPreferencesImpl writes.
_PREFS_HEADER = "<?xml version='1.0' encoding='utf-8' standalone='yes' ?>\n"
_INT32 = 2**31
# `run-as` refuses non-debuggable / unknown packages by printing, not by exiting loudly.
_RUN_AS_ERRORS = ("run-as:", "not debuggable", "is unknown", "permission denied", "inaccessible")
_BOOLEANS = frozenset({"true", "false"})


class Shell(Protocol):
    """The one device capability the read-back needs."""

    def shell(self, command: str) -> str: ...


class PrefsWriter(Protocol):
    """What a prefs WRITE needs on top of the read: private files, and app lifecycle.

    All four extras are semantic ``Device`` operations rather than native calls, so this stays
    a structural contract and this module never imports the Android runtime.
    """

    def shell(self, command: str) -> str: ...
    def read_app_file(self, package: str, path: str) -> bytes: ...
    def write_app_file(self, package: str, path: str, data: bytes) -> None: ...
    def remove_app_files(self, package: str, paths: list[str]) -> None: ...
    def stop_app(self, package: str) -> None: ...
    def launch_app(self, package: str, *, activity: str | None = None) -> None: ...


class PrefsSnapshot(NamedTuple):
    """The app's prefs file as it stood before a write, and where the write will land.

    Taken *after* the app is force-stopped, which is why it is a value rather than a re-read:
    a running process holds its own copy of every SharedPreferences file and flushes it on the
    way out, so anything read while the app lived is stale before it can be merged back.
    """

    package: str
    file: str  # the basename inside `shared_prefs`, never a path
    existed: bool
    xml: str | None

    @property
    def path(self) -> str:
        """Where the file lives, relative to the app's private data directory."""
        return f"shared_prefs/{self.file}"


class PrefsRead(NamedTuple):
    """What the app's own prefs say about the keys that were just requested."""

    applied: dict[str, str]
    ignored: list[str]
    mismatched: dict[str, str]
    files: list[str]
    reason: str | None  # why the read-back could not run (None = it ran)

    @property
    def verified(self) -> bool:
        return self.reason is None


class ContextPrefsRead(NamedTuple):
    """A privacy-filtered snapshot of feature flags already active in app prefs."""

    flags: dict[str, str]
    files: list[str]
    reason: str | None

    @property
    def verified(self) -> bool:
        return self.reason is None


def build_uri(package: str, pairs: dict[str, str], templates: dict[str, str] | None = None) -> str:
    """Build the set-flags deeplink for *package* from the configured templates."""
    tmpl = (templates or {}).get(package)
    if not tmpl:
        raise UsageError(
            f"no flags deeplink template for package {package!r}",
            hint=(
                "Set-flags schemes are app-specific, so there are no built-ins. Add one to "
                f'your config: `flags: {{templates: {{"{package}": '
                '"myapp://set-flags?{query}"}}`.'
            ),
        )
    if not pairs:
        raise UsageError("flags set needs at least one KEY=VAL")
    query = urlencode(pairs, quote_via=urlquote)
    if "{query}" in tmpl:
        return tmpl.replace("{query}", query)
    sep = "&" if "?" in tmpl else "?"
    return f"{tmpl}{sep}{query}"


def parse_assignments(items: list[str]) -> dict[str, str]:
    """Parse CLI ``KEY=VAL`` tokens."""
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise UsageError(
                f"flag assignment must be KEY=VAL, got {raw!r}",
                hint='e.g. `aua flags set <pkg> some_experiment=treatment_a`',
            )
        k, _, v = raw.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            raise UsageError(f"empty flag key in {raw!r}")
        out[k] = v
    return out


def load_flags_file(path: str | Path) -> tuple[str | None, dict[str, str]]:
    """Load a flags YAML: optional ``app:`` + ``flags:`` mapping (or bare mapping)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise UsageError(f"flags file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise UsageError(f"flags YAML does not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("flags file must be a mapping")
    app = data.get("app") or data.get("package")
    if "flags" in data:
        raw_flags = data.get("flags") or {}
    else:
        raw_flags = {k: v for k, v in data.items() if k not in ("app", "package")}
    if not isinstance(raw_flags, dict) or not raw_flags:
        raise UsageError("flags file needs a non-empty `flags:` mapping (or bare KEY: VAL)")
    cleaned = {str(k): "" if v is None else str(v) for k, v in raw_flags.items()}
    return (str(app) if app else None), cleaned


def prefs_dir(package: str) -> str:
    return f"/data/data/{package}/shared_prefs"


def parse_prefs(xml: str, keys: set[str]) -> dict[str, list[str]]:
    """Every value the prefs XML holds for *keys* (one key can appear in several files)."""
    found: dict[str, list[str]] = {}
    for m in _ENTRY_RE.finditer(xml):
        key = m.group("key")
        if key not in keys:
            continue
        value = m.group("attr")
        if value is None:
            value = m.group("text") or ""
        found.setdefault(key, []).append(value)
    return found


def parse_all_prefs(xml: str) -> dict[str, list[str]]:
    """Every primitive preference entry.

    Callers must filter the result before persisting it; app prefs can contain tokens
    and user data. Runtime context discovery only retains configured/flag-like keys.
    """
    found: dict[str, list[str]] = {}
    for match in _ENTRY_RE.finditer(xml):
        value = match.group("attr")
        if value is None:
            value = match.group("text") or ""
        found.setdefault(match.group("key"), []).append(value)
    return found


def _matches(want: str, got: str) -> bool:
    want, got = want.strip(), got.strip()
    if want == got:
        return True
    return want.lower() in _BOOLEANS and want.lower() == got.lower()


def _run_as_failure(output: str) -> str | None:
    """The line proving ``run-as`` could not read the app, or None when it worked."""
    for line in output.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if stripped and any(marker in low for marker in _RUN_AS_ERRORS):
            return stripped
    return None


def _run_as(device: Shell, package: str, argv: list[str]) -> tuple[str, str | None]:
    """Run *argv* as the app; return its output plus a reason when it could not run."""
    command = f"run-as {quote(package)} " + " ".join(quote(a) for a in argv)
    try:
        out = device.shell(command)
    except Exception as exc:  # noqa: BLE001 — every shell failure means "cannot verify"
        return "", f"{type(exc).__name__}: {exc}"
    return out, _run_as_failure(out)


def _candidate_files(device: Shell, package: str, keys: set[str], names: list[str]) -> list[str]:
    """Narrow the prefs files to those naming *keys*; the whole set when grep can't help."""
    directory = prefs_dir(package)
    paths = [f"{directory}/{name}" for name in names]
    if not paths or not all(_SAFE_KEY_RE.match(k) for k in keys):
        return paths
    pattern = 'name="(' + "|".join(sorted(keys)) + ')"'
    out, failed = _run_as(device, package, ["grep", "-lE", pattern, *paths])
    if failed:
        return paths
    return [line.strip() for line in out.splitlines() if line.strip().startswith(directory)]


def read_prefs(
    device: Shell, package: str, pairs: dict[str, str], *, prefs_file: str | None = None
) -> PrefsRead:
    """Read the app's prefs back and split *pairs* into applied / ignored / mismatched.

    Non-debuggable builds (and any other ``run-as`` refusal) come back carrying a
    ``reason`` instead of an answer, so the caller stops claiming the flags are set.
    """
    keys = set(pairs)
    directory = prefs_dir(package)
    listing, failed = _run_as(device, package, ["ls", directory])
    if failed:
        return PrefsRead({}, [], {}, [], failed)
    names = [n.strip() for n in listing.split() if n.strip().endswith(".xml")]
    if prefs_file is not None:
        names = [prefs_file] if prefs_file in names else []
        files = [f"{directory}/{name}" for name in names]
    else:
        files = _candidate_files(device, package, keys, names)
    xml = ""
    if files:
        xml, failed = _run_as(device, package, ["cat", *files])
        if failed:
            return PrefsRead({}, [], {}, [], failed)
    found = parse_prefs(xml, keys)
    applied: dict[str, str] = {}
    ignored: list[str] = []
    mismatched: dict[str, str] = {}
    for key, want in pairs.items():
        values = found.get(key, [])
        if any(_matches(want, got) for got in values):
            applied[key] = want
        elif values:
            mismatched[key] = values[0]
        else:
            ignored.append(key)
    return PrefsRead(applied, ignored, mismatched, [f.rsplit("/", 1)[-1] for f in files], None)


def read_context_flags(
    device: Shell,
    package: str,
    *,
    prefs_file: str | None = None,
    keys: list[str] | None = None,
    key_patterns: list[str] | None = None,
) -> ContextPrefsRead:
    """Read the current map-shaping flags without first writing a deeplink.

    Exact ``keys`` win. Otherwise only names matching ``key_patterns`` survive, so
    arbitrary app preferences are never copied into map memory.
    """
    directory = prefs_dir(package)
    listing, failed = _run_as(device, package, ["ls", directory])
    if failed:
        return ContextPrefsRead({}, [], failed)
    names = [name.strip() for name in listing.split() if name.strip().endswith(".xml")]
    if prefs_file is not None:
        names = [prefs_file] if prefs_file in names else []
        if not names:
            return ContextPrefsRead({}, [], f"prefs file not found: {prefs_file}")
    files = [f"{directory}/{name}" for name in names]
    if not files:
        return ContextPrefsRead({}, [], None)
    xml, failed = _run_as(device, package, ["cat", *files])
    if failed:
        return ContextPrefsRead({}, [], failed)

    entries = parse_all_prefs(xml)
    exact = set(keys or [])
    compiled: list[re.Pattern[str]] = []
    for raw in key_patterns or []:
        try:
            compiled.append(re.compile(raw))
        except re.error:
            continue

    def selected(key: str) -> bool:
        if exact:
            return key in exact
        return any(pattern.search(key) for pattern in compiled)

    flags = {
        key: values[-1]
        for key, values in entries.items()
        if values and selected(key)
    }
    return ContextPrefsRead(
        dict(sorted(flags.items())),
        [path.rsplit("/", 1)[-1] for path in files],
        None,
    )


def prefs_file_name(name: str) -> str:
    """Normalize a ``shared_prefs`` file reference to the basename Android stores.

    ``getSharedPreferences("settings")`` writes ``settings.xml``, and callers name the file
    either way, so the suffix is supplied rather than demanded. A path is refused instead: the
    write replaces whatever it addresses, and ``../databases/x`` addressed from the prefs
    directory is a different file in a different format.
    """
    value = str(name).strip()
    if value and not value.endswith(".xml"):
        value = f"{value}.xml"
    if not _PREFS_FILE_RE.fullmatch(value) or value.startswith("."):
        raise UsageError(
            f"invalid shared_prefs file name: {name!r}",
            hint="Pass the basename Android stores, e.g. `example_settings` or "
            "`example_settings.xml` — not a path.",
            code="prefs_file_invalid",
        )
    return value


def _prefs_entry(key: str, value: Any) -> ElementTree.Element:
    """One typed preference element, chosen by the Python type of *value*.

    The element name is not cosmetic: the app reads a given key with ``getBoolean`` or
    ``getInt``, and SharedPreferences throws ``ClassCastException`` when the stored type
    disagrees — so a boolean written as ``<string>true</string>`` crashes the reader instead of
    configuring it. Mapping YAML scalars onto the Android types leaves that decision with the
    flow author, who knows which getter the app calls.
    """
    if not isinstance(key, str) or not key.strip():
        raise UsageError("a shared_prefs key must be a non-empty string", code="prefs_key_invalid")
    if isinstance(value, bool):
        return ElementTree.Element("boolean", {"name": key, "value": "true" if value else "false"})
    if isinstance(value, int):
        # Android has two integer types and no widening after the fact: a value that does not
        # fit an int is stored as — and must be read as — a long.
        tag = "int" if -_INT32 <= value < _INT32 else "long"
        return ElementTree.Element(tag, {"name": key, "value": str(value)})
    if isinstance(value, float):
        return ElementTree.Element("float", {"name": key, "value": repr(value)})
    if isinstance(value, str):
        element = ElementTree.Element("string", {"name": key})
        element.text = value
        return element
    raise UsageError(
        f"unsupported shared_prefs value for {key!r}: {type(value).__name__}",
        hint="Values are strings, integers, floats, or booleans. String sets and deletions are "
        "not supported by this surface.",
        code="prefs_value_unsupported",
    )


def _prefs_text(value: Any) -> str:
    """The exact text the merged XML carries for *value* — what verification compares."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def merge_prefs_xml(xml: str | None, values: Mapping[str, Any]) -> str:
    """Return *xml* with *values* set, leaving every other entry as the app wrote it.

    A prefs file holds unrelated app state — session ids, counters, whatever the app keeps
    there — so a write is a merge, never a replacement: rebuilding the file from the requested
    keys alone would wipe state the flow never mentioned. Parsing rather than splicing text is
    what keeps entries this module does not model, such as ``<set>``, intact.
    """
    if xml and xml.strip():
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError as exc:
            raise UsageError(
                f"the existing shared_prefs XML does not parse: {exc}",
                hint="Refusing to overwrite a file AUA cannot read back; inspect it by hand.",
                code="prefs_unreadable",
            ) from exc
        if root.tag != "map":
            raise UsageError(
                f"shared_prefs XML has root <{root.tag}>, expected <map>",
                code="prefs_unreadable",
            )
    else:
        root = ElementTree.Element("map")
    for key, value in values.items():
        entry = _prefs_entry(str(key), value)
        stale = [child for child in root if child.get("name") == key]
        if stale:
            at = list(root).index(stale[0])
            for child in stale:
                root.remove(child)
            root.insert(at, entry)
        else:
            root.append(entry)
    ElementTree.indent(root, space="    ")
    return _PREFS_HEADER + ElementTree.tostring(root, encoding="unicode") + "\n"


def snapshot_prefs(device: PrefsWriter, package: str, file: str) -> PrefsSnapshot:
    """Force-stop *package* and read the prefs file a write is about to replace.

    **The force-stop is part of the contract, not an optimization.** A running process holds
    its own copy of every SharedPreferences file and writes it back when it is backgrounded or
    killed, so a write made underneath a live app is silently reverted later: the call reports
    success and the preference disappears. Stopping first is also why the snapshot is taken
    here rather than by the caller — the app flushes as it dies, so anything read while it was
    alive is already out of date.

    Separated from :func:`write_prefs` so the caller can journal an undo between the two: the
    restore point has to exist, and be recorded, before the file is touched.
    """
    package = package.strip()
    if not package:
        raise UsageError("a prefs write needs a package")
    name = prefs_file_name(file)
    directory = prefs_dir(package)
    listing, failed = _run_as(device, package, ["ls", directory])
    if failed:
        raise DeviceError(
            f"cannot write {package} preferences: {failed}",
            hint="Preference writes need an installed debuggable build; Android run-as refuses "
            "production builds.",
            code="prefs_access",
        )
    existed = name in {entry.strip() for entry in listing.split() if entry.strip()}
    try:
        device.stop_app(package)
    except Exception as exc:
        raise DeviceError(
            f"could not stop {package} before writing its preferences: {exc}",
            hint="An unstopped app overwrites the file from memory, so the write is not "
            "attempted at all.",
            code="prefs_stop_failed",
        ) from exc
    xml: str | None = None
    if existed:
        try:
            xml = device.read_app_file(package, f"shared_prefs/{name}").decode(
                "utf-8", errors="replace"
            )
        except Exception as exc:
            raise DeviceError(
                f"could not read {package}/shared_prefs/{name}: {exc}",
                code="prefs_read_failed",
            ) from exc
    return PrefsSnapshot(package=package, file=name, existed=existed, xml=xml)


def prefs_backup_path(cache_dir: str | Path, serial: str, package: str, file: str) -> Path:
    """Where the pre-write copy of one app's prefs file lives on the host."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{serial}-{package}-{file}")
    return Path(cache_dir).expanduser() / "prefs" / f"{safe}.json"


def save_prefs_backup(cache_dir: str | Path, serial: str, snapshot: PrefsSnapshot) -> Path:
    """Persist *snapshot* as the restore point the undo ledger will replay.

    Host-side and outside the process that wrote it, because the agent that changed the
    preference is exactly the process that may not survive to change it back — and a build
    left pointing at a staging backend is inherited by whoever picks the device up next.
    """
    path = prefs_backup_path(cache_dir, serial, snapshot.package, snapshot.file)
    atomic_write_text(
        path,
        json.dumps(
            {
                "package": snapshot.package,
                "file": snapshot.file,
                "existed": snapshot.existed,
                "xml": snapshot.xml,
            },
            indent=2,
        )
        + "\n",
    )
    return path


def restore_prefs(device: PrefsWriter, backup_path: str | Path) -> str:
    """Put the prefs file back the way :func:`save_prefs_backup` found it.

    A file that did not exist before is removed rather than blanked: an empty ``<map/>`` is not
    the same state as "the app has never written this file", and some apps seed defaults only
    on that first miss. The app is stopped first for the same reason the write stops it, and is
    deliberately *not* relaunched — a teardown that starts apps is a teardown that surprises
    whoever inherits the device.
    """
    path = Path(backup_path).expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(
            f"the shared_prefs restore point is unreadable: {path}",
            hint="Inspect the app's own prefs by hand; AUA will not guess at their contents.",
            code="prefs_backup_unreadable",
        ) from exc
    package = str(raw.get("package") or "")
    name = prefs_file_name(str(raw.get("file") or ""))
    if not package:
        raise UsageError(
            f"the shared_prefs restore point names no package: {path}",
            code="prefs_backup_unreadable",
        )
    device.stop_app(package)
    target = f"shared_prefs/{name}"
    if not raw.get("existed"):
        device.remove_app_files(package, [target])
        return f"{package}/{name} removed (it did not exist before)"
    xml = raw.get("xml")
    if not isinstance(xml, str):
        raise UsageError(
            f"the shared_prefs restore point for {package}/{name} holds no XML: {path}",
            code="prefs_backup_unreadable",
        )
    device.write_app_file(package, target, xml.encode("utf-8"))
    return f"{package}/{name} restored"


def write_prefs(
    device: PrefsWriter,
    snapshot: PrefsSnapshot,
    values: Mapping[str, Any],
    *,
    relaunch: bool = True,
) -> dict[str, Any]:
    """Merge *values* into the prefs file *snapshot* describes, then verify them on disk.

    The app is already stopped (see :func:`snapshot_prefs`) and is relaunched afterwards by
    default, because cold start is when it re-reads the file — the same reason ``flags set``
    restarts it. Pass ``relaunch=False`` when a flow wants to arrange several things before
    the app comes up.

    Verification re-reads the file from the device instead of trusting the bytes that were
    sent, which is what separates "a file was written" from "the preference is set".
    """
    if not values:
        raise UsageError(f"a prefs write needs at least one key for {snapshot.package}")
    package, name, path = snapshot.package, snapshot.file, snapshot.path
    merged = merge_prefs_xml(snapshot.xml, values)
    if not snapshot.existed:
        # An app that has never written a preference has no `shared_prefs` directory at all, and
        # the write would then fail on the missing parent — a failure the caller cannot act on.
        # Created as the app uid, so ownership and SELinux context are the app's own.
        _run_as(device, package, ["mkdir", "-p", prefs_dir(package)])
    try:
        device.write_app_file(package, path, merged.encode("utf-8"))
    except Exception as exc:
        raise DeviceError(
            f"could not write {package}/{path}: {exc}",
            code="prefs_write_failed",
        ) from exc
    written = {str(key): _prefs_text(value) for key, value in values.items()}
    verify_error: str | None = None
    present: dict[str, list[str]] = {}
    try:
        present = parse_all_prefs(
            device.read_app_file(package, path).decode("utf-8", errors="replace")
        )
    except Exception as exc:  # noqa: BLE001 — an unverifiable write is reported, not raised
        verify_error = f"{type(exc).__name__}: {exc}"
    lost = (
        []
        if verify_error is not None
        else sorted(
            key
            for key, want in written.items()
            if not any(_matches(want, got) for got in present.get(key, []))
        )
    )
    relaunched = False
    if relaunch:
        try:
            device.launch_app(package)
            relaunched = True
        except Exception as exc:
            raise DeviceError(
                f"preferences written but {package} could not be relaunched: {exc}",
                hint=f"Relaunch it explicitly with `aua app launch {package}`.",
                code="prefs_restart_failed",
            ) from exc
    payload: dict[str, Any] = {
        "ok": verify_error is None and not lost,
        "action": "prefs-write",
        "package": package,
        "file": name,
        "created": not snapshot.existed,
        "written": written,
        "verified": verify_error is None and not lost,
        "relaunched": relaunched,
        "detail": f"{len(written)} preference(s) written to {name}",
    }
    if verify_error is not None:
        payload["verify_error"] = verify_error
        payload["detail"] = f"{name} written but unverified: {verify_error}"
    elif lost:
        payload["lost"] = lost
        payload["detail"] = f"{name} did not keep: {', '.join(lost)}"
    return payload


def dump_result(
    *,
    package: str,
    uri: str,
    flags: dict[str, str],
    prefs: PrefsRead | None = None,
    restarted: bool = False,
    activity: str | None = None,
    restart_error: str | None = None,
) -> dict[str, Any]:
    """The answer: what was asked, what landed, and whether the app came back up."""
    payload: dict[str, Any] = {
        "ok": True,
        "action": "flags-set",
        "package": package,
        "uri": uri,
        "flags": flags,
        "verified": prefs is not None and prefs.verified,
        "restarted": restarted,
    }
    if activity:
        payload["activity"] = activity
    if restart_error:
        payload["restart_error"] = restart_error
    problems = [restart_error] if restart_error else []
    if prefs is None:
        payload["detail"] = f"{uri} (unverified: verification off)"
    elif prefs.reason is not None:
        payload["verify_error"] = prefs.reason
        payload["detail"] = f"{uri} (unverified: {prefs.reason})"
    else:
        payload["applied"] = prefs.applied
        payload["ignored"] = prefs.ignored
        payload["prefs"] = prefs.files
        if prefs.mismatched:
            payload["mismatched"] = prefs.mismatched
        lost = prefs.ignored + sorted(prefs.mismatched)
        if lost:
            problems.insert(
                0, f"{len(lost)} of {len(flags)} flags are not set on device: {', '.join(lost)}"
            )
        else:
            payload["detail"] = f"{uri} ({len(prefs.applied)} flags verified on device)"
    if problems:
        payload["ok"] = False
        payload["detail"] = "; ".join(problems)
    return payload


__all__ = [
    "ContextPrefsRead",
    "PrefsRead",
    "PrefsSnapshot",
    "build_uri",
    "dump_result",
    "load_flags_file",
    "merge_prefs_xml",
    "parse_assignments",
    "parse_all_prefs",
    "parse_prefs",
    "prefs_backup_path",
    "prefs_dir",
    "prefs_file_name",
    "read_context_flags",
    "read_prefs",
    "restore_prefs",
    "save_prefs_backup",
    "snapshot_prefs",
    "write_prefs",
]
