"""Feature-flag deeplink helper — apply KEY=VAL via a package-specific URI template.

No SharedPreferences / run-as. Luzia default: ``luzia-test://set-flags?k=v&…``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import yaml

from .errors import UsageError

# package → URI template with ``{query}`` placeholder (already encoded key=val&…)
DEFAULT_TEMPLATES: dict[str, str] = {
    "co.thewordlab.luzia.dev": "luzia-test://set-flags?{query}",
    "co.thewordlab.luzia": "luzia-test://set-flags?{query}",
}


def build_uri(package: str, pairs: dict[str, str], templates: dict[str, str] | None = None) -> str:
    """Build the set-flags deeplink for *package*."""
    tmpl = (templates or {}).get(package) or DEFAULT_TEMPLATES.get(package)
    if not tmpl:
        raise UsageError(
            f"no flags deeplink template for package {package!r}",
            hint="Pass a template via config flags.templates or use a known Luzia package.",
        )
    if not pairs:
        raise UsageError("flags set needs at least one KEY=VAL")
    query = urlencode(pairs, quote_via=quote)
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
                hint='e.g. `aua flags set <pkg> apps_hub_experiment=a`',
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


def dump_result(*, package: str, uri: str, flags: dict[str, str]) -> dict[str, Any]:
    return {"ok": True, "action": "flags-set", "package": package, "uri": uri, "flags": flags}


__all__ = [
    "DEFAULT_TEMPLATES",
    "build_uri",
    "dump_result",
    "load_flags_file",
    "parse_assignments",
]
