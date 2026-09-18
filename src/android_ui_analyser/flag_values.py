"""Platform-neutral feature-flag values, configuration and result formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote as urlquote
from urllib.parse import urlencode

import yaml

from .errors import UsageError


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
                hint="e.g. `aua flags set <pkg> some_experiment=treatment_a`",
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
