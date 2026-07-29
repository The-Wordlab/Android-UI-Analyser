"""Feature-flag helper: apply KEY=VAL via a package-specific deeplink, then read it back.

Templates come from user config (``flags.templates``) only — a set-flags scheme is an
app's private contract, and firing one that no installed build declares is
indistinguishable from success. The read-back is what turns "a deeplink was fired" into
"these keys are set": on a debuggable build the overrides are readable straight out of the
app's own ``shared_prefs`` XML.
"""

from __future__ import annotations

import re
from pathlib import Path
from shlex import quote
from typing import Any, NamedTuple, Protocol
from urllib.parse import quote as urlquote
from urllib.parse import urlencode

import yaml

from .errors import UsageError

# One prefs entry, attribute-valued (`<boolean … value="true" />`) or text-valued
# (`<string …>a</string>`); the backreference stops the two shapes from crossing over.
_ENTRY_RE = re.compile(
    r'<(string|boolean|int|long|float)\s+name="(?P<key>[^"]+)"'
    r'(?:\s+value="(?P<attr>[^"]*)"\s*/>|\s*>(?P<text>[^<]*)</\1>)'
)
_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
# `run-as` refuses non-debuggable / unknown packages by printing, not by exiting loudly.
_RUN_AS_ERRORS = ("run-as:", "not debuggable", "is unknown", "permission denied", "inaccessible")
_BOOLEANS = frozenset({"true", "false"})


class Shell(Protocol):
    """The one device capability the read-back needs."""

    def shell(self, command: str) -> str: ...


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
    "PrefsRead",
    "build_uri",
    "dump_result",
    "load_flags_file",
    "parse_assignments",
    "parse_prefs",
    "prefs_dir",
    "read_prefs",
]
