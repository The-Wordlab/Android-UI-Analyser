"""Versioned API maps: what each client version sends its backend and needs back.

An agent fills the map from a client's source at a release tag, one folder per version:

    <map>/<client>/<version>/version.yaml   tag, commit, released; what has been frozen
    <map>/<client>/<version>/schema.yaml    every endpoint this version calls, and what it requires
    <map>/<client>/<version>/flows/*.yaml   cartridges: one user flow's calls, in order

A released version's code never changes, so whatever its map says about an endpoint or a flow
stays true: freezing records each entry, and a later edit to one is reported. The map can still
grow, one screen at a time, because new entries are not edits. The next release starts from the
nearest older folder (`--from`) and changes only what the diff between the two tags touched. AUA runs no model here: the calling agent reads the source and writes the
files, and AUA finds tags, diffs versions, reports what is still unmapped and checks responses
against the version's schema (`aua api run`).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .api_spec_diff import normalize_path, operation_key
from .api_usage import _fetch, _git, _rev, find_release, load_config, scan_client
from .atomic import atomic_write_text
from .errors import UsageError

_VERSION_FILE = "version.yaml"
_SCHEMA_FILE = "schema.yaml"
_FLOWS = "flows"
_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")
_SCHEMA_SKELETON = """\
# What this client version needs back from each endpoint it calls, written from its data models
# at this version's tag. Paths are relative to the flows' base_url.
#   name: type    required: missing or null means this version cannot decode the response
#   name?: type   optional: may be absent or null (absent ones are listed, never flagged)
#   name: type?   must be present, but may be null
# Inside a one-line {…} mapping, quote anything with a `?`: {"name?": string, b: "string?"}.
# Types: string, number, boolean, any, enum(a, b), {nested: object}, [element].
# A streamed response lists its events instead of a body: `events: {done: {...}, tool?: {...}}`,
# where `done` must arrive in every stream and `tool?` is only decoded when it is sent.
endpoints: {}
#  "POST /v1/notes":
#    response:
#      body:
#        id: string
#        pinned?: boolean
#        state: enum(draft, published)
#        tags: [string]
"""


def endpoint_key(method: str, path: str) -> str:
    """``POST /notes/{{id}}`` and ``POST /notes/{id}`` both become ``POST /notes/{}``."""
    return operation_key(method, normalize_path(_TEMPLATE.sub("{x}", path)))


def _client(config: dict[str, Any], name: str) -> dict[str, Any]:
    for client in config["clients"]:
        if client["name"] == name:
            return dict(client)
    known = ", ".join(c["name"] for c in config["clients"])
    raise UsageError(f"no client named {name!r}", hint=f"Clients in {config['file']}: {known}")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def entries(folder: Path) -> dict[str, str]:
    """A hash per schema endpoint (of its content, not its comments) and per flow file."""
    found = {
        f"endpoint {key}": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        for key, contract in schema_endpoints(folder).items()
    }
    for flow in sorted((folder / _FLOWS).glob("*.yaml")):
        found[f"flow {flow.stem}"] = hashlib.sha256(flow.read_bytes()).hexdigest()
    return found


def drift(folder: Path, frozen: Any) -> dict[str, list[str]]:
    """Frozen entries that changed or vanished, and entries added since the last freeze."""
    frozen = frozen if isinstance(frozen, dict) else {}
    now = entries(folder)
    return {
        "changed": sorted(k for k, digest in frozen.items() if now.get(k) != digest),
        "added": sorted(k for k in now if k not in frozen),
    }


def schema_endpoints(folder: Path) -> dict[str, dict[str, Any]]:
    """The version's schema as ``{"POST /notes/{}": {"response": …}}``."""
    file = folder / _SCHEMA_FILE
    try:
        doc = yaml.safe_load(file.read_text(encoding="utf-8")) if file.is_file() else {}
    except yaml.YAMLError as exc:
        # Never read a broken schema as an empty one: every check would silently pass.
        raise UsageError(
            f"{file} is not valid YAML: {str(exc).splitlines()[0]}",
            hint="Inside a one-line {…} mapping, quote anything with a `?`: "
            '{"name?": string, other: "string?"}; or write that mapping one key per line.',
        ) from exc
    raw = (doc or {}).get("endpoints") or {} if isinstance(doc, dict) else {}
    if not isinstance(raw, dict):
        raise UsageError(f"`endpoints` in {folder / _SCHEMA_FILE} must be a mapping")
    out: dict[str, dict[str, Any]] = {}
    for key, contract in raw.items():
        method, _, path = str(key).partition(" ")
        if not path.startswith("/"):
            raise UsageError(f"schema endpoint {key!r} must look like `GET /path`")
        out[endpoint_key(method, path)] = contract if isinstance(contract, dict) else {}
    return out


def version_context(flow: str | Path) -> dict[str, Any] | None:
    """The map version a flow file belongs to, or None for a cartridge outside any map."""
    folder = Path(flow).resolve().parent.parent
    meta = _read_yaml(folder / _VERSION_FILE)
    if Path(flow).resolve().parent.name != _FLOWS or not meta.get("version"):
        return None
    frozen = meta.get("frozen")
    context: dict[str, Any] = {
        "client": meta.get("client"),
        "version": meta.get("version"),
        "tag": meta.get("tag"),
        "released": bool(meta.get("released")),
        "frozen": bool(frozen),
        "schema": schema_endpoints(folder),
    }
    if frozen:
        context["changed_since_freeze"] = drift(folder, frozen)["changed"]
    return context


def _relative(path: str, base: str) -> str:
    """A scanned path, which includes the client's base path, as the map spells it."""
    if base not in ("", "/") and (path == base or path.startswith(base + "/")):
        return path[len(base) :] or "/"
    return path


def _calls(scan: dict[str, Any], label: str, base: str) -> dict[str, str]:
    """``{"POST /notes/{}": "src/NotesApi.kt:12"}`` for one scanned version."""
    out: dict[str, str] = {}
    for call in scan["calls"].get(label, []):
        key = operation_key(call["method"], _relative(call["path"], base))
        out.setdefault(key, f"{call['file']}:{call['line']}")
    return out


def _uncovered(calls: dict[str, str], schema: dict[str, Any]) -> list[str]:
    """Calls with no schema entry. A call whose method the scanner could not read is `* /path`."""
    paths = {key.split(" ", 1)[1] for key in schema}
    return sorted(
        key
        for key in calls
        if key not in schema and not (key.startswith("* ") and key.split(" ", 1)[1] in paths)
    )


def _base(client: dict[str, Any]) -> str:
    return normalize_path(str(client.get("base_path") or "/"))


def status(
    config_path: str | Path,
    cache_dir: str | Path,
    *,
    only: str | None = None,
    fetch: bool = True,
) -> dict[str, Any]:
    """Every mapped version, whether it is complete and untouched, and what is still unmapped."""
    config = load_config(config_path)
    root = Path(config["map"])
    clients = [_client(config, only)] if only else [dict(c) for c in config["clients"]]
    report: list[dict[str, Any]] = []
    for client in clients:
        folders = sorted(
            p for p in (root / client["name"]).glob("*") if (p / _VERSION_FILE).is_file()
        )
        metas = [(folder, _read_yaml(folder / _VERSION_FILE)) for folder in folders]
        extra = [
            {"label": str(meta["version"]), "ref": meta.get("tag"), "sha": meta.get("commit")}
            for _, meta in metas
            if meta.get("version") and meta.get("commit")
        ]
        scan = scan_client(client, Path(cache_dir) / "api-usage", fetch=fetch, extra_refs=extra)
        base = _base(client)
        versions = []
        for folder, meta in metas:
            label = str(meta.get("version"))
            schema = schema_endpoints(folder)
            calls = _calls(scan, label, base)
            entry: dict[str, Any] = {
                "version": label,
                "tag": meta.get("tag"),
                "commit": str(meta.get("commit") or "")[:12],
                "released": bool(meta.get("released")),
                "frozen": bool(meta.get("frozen")),
                "flows": sorted(p.stem for p in (folder / _FLOWS).glob("*.yaml")),
                "schema_endpoints": len(schema),
                "calls_not_in_schema": _uncovered(calls, schema),
                "schema_not_called": sorted(k for k in schema if k not in calls),
            }
            if meta.get("frozen"):
                moved = drift(folder, meta["frozen"])
                entry["changed_since_freeze"] = moved["changed"]
                entry["added_since_freeze"] = moved["added"]
            if not meta.get("released") and meta.get("tag"):
                now = _rev(client["repo"], str(meta["tag"]))
                entry["moved"] = bool(now and now != meta.get("commit"))
            versions.append(entry)
        mapped = {v["version"] for v in versions}
        report.append(
            {
                "name": client["name"],
                "folder": str(root / client["name"]),
                "versions": versions,
                "releases_not_mapped": [
                    ref["label"]
                    for ref in scan["refs"]
                    if ref["label"] != "unreleased" and ref["label"] not in mapped
                ],
            }
        )
    return {"ok": True, "map": str(root), "clients": report}


def new_version(
    config_path: str | Path,
    cache_dir: str | Path,
    client_name: str,
    version: str,
    *,
    ref: str | None = None,
    from_version: str | None = None,
    fetch: bool = True,
) -> dict[str, Any]:
    """Create one version's folder, and say what the agent has to read to fill it."""
    config = load_config(config_path)
    client = _client(config, client_name)
    repo = client["repo"]
    root = Path(config["map"]) / client_name
    folder = root / version
    if folder.exists():
        raise UsageError(
            f"{client_name} {version} is already mapped at {folder}",
            hint="Edit its files, or freeze it once complete: `aua api map freeze`.",
        )
    if fetch:
        _fetch(repo)  # offline is fine: map what this clone already has
    if ref:
        sha = _rev(repo, ref)
        if sha is None:
            raise UsageError(f"{ref!r} does not resolve in {repo}")
        release = {"label": version, "ref": ref, "sha": sha}
    else:
        found = find_release(client, version)
        if found is None:
            raise UsageError(
                f"no {client_name} release tag has version {version}",
                hint="Check the client's `tags` and `version` rule in the config. For code that "
                "is not released, pass `--ref <branch or commit>`.",
            )
        release = found
    source = root / from_version if from_version else None
    if source is not None and not (source / _VERSION_FILE).is_file():
        raise UsageError(f"{client_name} {from_version} is not mapped, so it cannot be copied")

    refs: list[dict[str, Any]] = [release]
    if source is not None:
        old = _read_yaml(source / _VERSION_FILE)
        refs.append({"label": str(from_version), "ref": old.get("tag"), "sha": old.get("commit")})
    scan = scan_client(client, Path(cache_dir) / "api-usage", fetch=False, extra_refs=refs)
    base = _base(client)
    calls = _calls(scan, version, base)

    folder.mkdir(parents=True)
    if source is not None:
        for item in source.iterdir():
            if item.name == _VERSION_FILE:
                continue
            target = folder / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    else:
        atomic_write_text(folder / _SCHEMA_FILE, _SCHEMA_SKELETON)
    (folder / _FLOWS).mkdir(exist_ok=True)
    meta = {
        "client": client_name,
        "version": version,
        "tag": release["ref"],
        "commit": release["sha"],
        "released": ref is None,
        "created": datetime.now(UTC).date().isoformat(),
        "copied_from": from_version,
        "frozen": None,
    }
    atomic_write_text(folder / _VERSION_FILE, yaml.safe_dump(meta, sort_keys=False))

    result: dict[str, Any] = {
        "ok": True,
        "client": client_name,
        "version": version,
        "tag": release["ref"],
        "commit": release["sha"],
        "path": str(folder),
        "calls": [{"call": key, "source": src} for key, src in sorted(calls.items())],
    }
    if source is not None:
        before = _calls(scan, str(from_version), base)
        old_sha = str(refs[1]["sha"])
        changed = _git(repo, "diff", "--name-only", old_sha, release["sha"]).splitlines()
        api_files = {s.rsplit(":", 1)[0] for s in [*calls.values(), *before.values()]}
        include = list(client.get("files") or ["*"])
        result["changes_since"] = {
            "version": from_version,
            "calls_added": [
                {"call": k, "source": calls[k]} for k in sorted(calls.keys() - before.keys())
            ],
            "calls_removed": sorted(before.keys() - calls.keys()),
            "api_files_changed": sorted(f for f in changed if f in api_files),
            "source_files_changed": len(
                [f for f in changed if any(fnmatch.fnmatch(f, g) for g in include)]
            ),
        }
    result["next"] = (
        f"Read {client_name}'s code at {release['ref']} without switching any checkout "
        f"(`git -C {repo} show {release['ref']}:<file>`, or a temporary worktree). For each call "
        f"above, write what this version requires back into {folder / _SCHEMA_FILE}, and the user "
        f"flows that matter into {folder / _FLOWS}/. Run them with `aua api run`; once they pass "
        f"against a backend this version works with, `aua api map freeze {client_name} {version}`."
    )
    return result


def freeze(
    config_path: str | Path, client_name: str, version: str, *, force: bool = False
) -> dict[str, Any]:
    """Record every entry's fingerprint, so a later edit to a released version's map shows.

    New entries are frozen alongside the old ones; a changed frozen entry needs ``force``.
    """
    config = load_config(config_path)
    _client(config, client_name)
    folder = Path(config["map"]) / client_name / version
    meta = _read_yaml(folder / _VERSION_FILE)
    if not meta:
        raise UsageError(f"{client_name} {version} is not mapped")
    if not meta.get("released"):
        raise UsageError(
            f"{client_name} {version} is not a release, and unreleased code keeps moving",
            hint="Freeze the folder of a tagged release instead.",
        )
    if not schema_endpoints(folder) or not list((folder / _FLOWS).glob("*.yaml")):
        raise UsageError(
            f"{client_name} {version} has no schema endpoints or no flows yet",
            hint="Freeze a version once its schema and flows are written and pass.",
        )
    moved = drift(folder, meta.get("frozen"))
    if moved["changed"] and not force:
        raise UsageError(
            f"{client_name} {version} changed after it was frozen: {', '.join(moved['changed'])}",
            hint="A released version's code never changes, so what its map says should not "
            "either. If an entry was wrong, fix it and pass --force to freeze it again.",
        )
    meta["frozen"] = entries(folder)
    atomic_write_text(folder / _VERSION_FILE, yaml.safe_dump(meta, sort_keys=False))
    result: dict[str, Any] = {
        "ok": True,
        "client": client_name,
        "version": version,
        "frozen": len(meta["frozen"]),
        "added": moved["added"],
    }
    if moved["changed"]:
        result["refrozen"] = moved["changed"]
    return result
