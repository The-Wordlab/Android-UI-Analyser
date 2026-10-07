"""Which client versions call which backend endpoint, and what a backend change touches.

Device-less. Client repositories are read through git at any ref, with no checkout, and the
backend's OpenAPI spec is exported at two points and diffed (:mod:`api_spec_diff`). Every
backend change is reported beside the client versions and source lines that call the changed
operation. AUA finds the candidates; the calling agent reads those lines and decides whether
the client really depends on what changed — :data:`AGENT_BRIEF` is the instruction it gets.

Nothing is kept up to date in the background. Each run fetches, scans only file versions it
has never seen (keyed by git blob id, so an unchanged file is read once across every release),
and answers for the history as it stands at that moment. The config names the repositories
and how each one spells a call; nothing about any particular app lives in this module.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .api_spec_diff import (
    diff_specs,
    match_operations,
    normalize_path,
    operation_key,
    operations,
    untyped_response,
)
from .atomic import atomic_write_text
from .errors import UsageError

AGENT_BRIEF = (
    "Each finding is a backend change that the listed client versions may feel. AUA matched "
    "routes, not the fields each client reads, so classify every finding for every listed "
    "client version:\n"
    "- decode_fails: the version's model cannot take the new shape — a non-optional field "
    "removed or now null, a type it cannot parse, a strict enum meeting a new value, or a "
    "request the server now rejects.\n"
    "- data_missing: decoding succeeds but data or behaviour is silently lost — an optional or "
    "defaulted field now absent, null or renamed, so the screen shows nothing or the default.\n"
    "- unaffected: the version's model does not declare the field, or already reads the new "
    "spelling and type.\n"
    "- unknown: say what you could not see (custom decoder, generated code, decoder settings).\n"
    "Read each version's own sources with `git -C <repo> show <sha>:<file>` and follow the call "
    "to the model it decodes or the request it builds; versions differ, so never infer one from "
    "another. Decoder settings decide null and unknown-value handling (kotlinx `explicitNulls` "
    "and `coerceInputValues`, Moshi on null for a non-null property, Swift "
    "`keyDecodingStrategy`). Cite the model line. `unused_changes` are changes no scanned client "
    "version calls, which is not proof they are safe: see `coverage.not_checked`."
)

NOT_CHECKED = [
    "calls the client patterns do not match: URLs built from constants, concatenation or at "
    "runtime",
    "clients, backends and versions this config does not list (each client's `history` says "
    "which tags were left out)",
    "fields of the operations in `untyped_called_operations`: only removing them shows",
    "authentication, headers, status codes, streaming payloads and anything else the OpenAPI "
    "spec does not describe",
]

CONFIG_HINT = (
    "Write an api-usage config (YAML or JSON), e.g.\n"
    "backend:\n"
    "  repo: .\n"
    "  base: origin/main\n"
    "  spec_command: python scripts/export_openapi.py {out}\n"
    "clients:\n"
    "  - name: android\n"
    "    repo: ../example-android\n"
    "    scanner: retrofit\n"
    "    base_path: /api/v1/\n"
    "    tags: 'v*'\n"
    "  - name: web\n"
    "    repo: ../example-web\n"
    "    files: ['src/*']\n"
    "    pattern: '(?P<path>/api/v1(?:/[A-Za-z0-9_\\-{}$]+)+)'"
)

_RETROFIT = [
    r'@(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\(\s*(?:value\s*=\s*)?"(?P<path>[^"]*)"',
    # `@HTTP(method = "DELETE", path = "…", hasBody = true)` — the only way Retrofit sends a
    # DELETE with a body; the lookaheads accept the arguments in either order.
    r'@HTTP\((?=[^)]*\bmethod\s*=\s*"(?P<method>[A-Za-z]+)")(?=[^)]*\bpath\s*=\s*"(?P<path>[^"]*)")',
]
_SCANNERS: dict[str, dict[str, Any]] = {
    "retrofit": {"pattern": _RETROFIT, "files": ["*.kt", "*.java"]},
}
_DEFAULT_MAX_VERSIONS = 10
_READ_CHUNK = 2000
_GIT_TIMEOUT = 300
_UNMATCHED_SHOWN = 25
# Reads never fetch one object at a time; see `_prefetch`.
_NO_LAZY_FETCH = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}


# ---------------------------------------------------------------------------------- config


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the config; relative repo paths resolve against the config's own folder."""

    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise UsageError(f"no api-usage config at {file}", hint=CONFIG_HINT)
    raw = yaml.safe_load(file.read_text()) or {}
    backend = dict(raw.get("backend") or {})
    clients = [dict(c) for c in raw.get("clients") or []]
    if not backend.get("repo") or not clients:
        raise UsageError(
            "an api-usage config needs `backend.repo` and at least one client", hint=CONFIG_HINT
        )
    here = file.parent
    backend["repo"] = str((here / str(backend["repo"])).expanduser().resolve())
    for client in clients:
        if not client.get("name") or not client.get("repo"):
            raise UsageError("every client needs `name` and `repo`", hint=CONFIG_HINT)
        client["repo"] = str((here / str(client["repo"])).expanduser().resolve())
        scanner = client.get("scanner")
        if scanner:
            if scanner not in _SCANNERS:
                raise UsageError(
                    f"unknown scanner {scanner!r} for client {client['name']}",
                    hint=f"Known scanners: {', '.join(sorted(_SCANNERS))}; or give `pattern`.",
                )
            for key, value in _SCANNERS[scanner].items():
                client.setdefault(key, value)
        if not client.get("pattern") or not client.get("files"):
            raise UsageError(
                f"client {client['name']} needs `scanner`, or both `pattern` and `files`",
                hint=CONFIG_HINT,
            )
        # One pattern, or a list of them for a client that spells a call more than one way.
        if isinstance(client["pattern"], str):
            client["pattern"] = [client["pattern"]]
        if any("(?P<path>" not in str(p) for p in client["pattern"]):
            raise UsageError(f"client {client['name']}: every `pattern` needs a named group `path`")
    # Where `aua api map` keeps each client version's schema and flows.
    api_map = str((here / str(raw.get("map") or "aua-map")).expanduser().resolve())
    return {"file": str(file), "backend": backend, "clients": clients, "map": api_map}


# ------------------------------------------------------------------------------------- git


def _git(repo: str, *args: str, timeout: int = _GIT_TIMEOUT) -> str:
    proc = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip()[:400] or f"exit {proc.returncode}"
        raise UsageError(f"git {' '.join(args[:2])} failed in {repo}: {detail}")
    return proc.stdout


def _rev(repo: str, ref: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() or None


def _fetch(repo: str) -> str:
    try:
        _git(repo, "fetch", "--quiet", "--tags", "origin", timeout=120)
    except (UsageError, subprocess.TimeoutExpired) as exc:
        return f"failed, used local refs: {str(exc)[:200]}"
    return "ok"


def _prefetch(repo: str, shas: list[str]) -> None:
    """Fetch the blobs a partial clone lacks in one request, instead of one request per file.

    A ``--filter=blob:none`` clone holds no old file contents, and reading one makes git fetch it
    alone — a history scan of one mobile repo spent over ten minutes doing that.
    """

    remote = subprocess.run(
        ["git", "-C", repo, "config", "--get", "extensions.partialclone"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not remote:
        return
    # A second pass: measured against a real remote, one request brought 5302 of 5427 blobs and
    # the retry brought the rest.
    for _ in range(2):
        listing = subprocess.run(
            ["git", "-C", repo, "cat-file", "--batch-check=%(objectname)"],
            input="\n".join(shas) + "\n",
            capture_output=True,
            text=True,
            env=_NO_LAZY_FETCH,
            timeout=_GIT_TIMEOUT,
        ).stdout
        missing = [line.split()[0] for line in listing.splitlines() if line.endswith(" missing")]
        if not missing:
            return
        subprocess.run(
            ["git", "-C", repo, "-c", "fetch.negotiationAlgorithm=noop", "fetch", remote]
            + ["--no-tags", "--no-write-fetch-head", "--recurse-submodules=no"]
            + ["--filter=blob:none", "--stdin"],
            input="\n".join(missing) + "\n",
            capture_output=True,
            text=True,
            env=_NO_LAZY_FETCH,
            timeout=_GIT_TIMEOUT,
        )


def _read_blobs(repo: str, shas: list[str]) -> dict[str, str]:
    """Blob contents by id; a blob that is still missing after :func:`_prefetch` is skipped."""

    out: dict[str, str] = {}
    for start in range(0, len(shas), _READ_CHUNK):
        chunk = shas[start : start + _READ_CHUNK]
        data = subprocess.run(
            ["git", "-C", repo, "cat-file", "--batch"],
            input=("\n".join(chunk) + "\n").encode(),
            capture_output=True,
            check=True,
            env=_NO_LAZY_FETCH,
            timeout=_GIT_TIMEOUT,
        ).stdout
        at = 0
        for sha in chunk:
            end = data.index(b"\n", at)
            header = data[at:end].split()
            at = end + 1
            if len(header) < 3 or header[1] == b"missing":
                continue
            size = int(header[2])
            out[sha] = data[at : at + size].decode("utf-8", "replace")
            at += size + 1
    return out


def _files_at(repo: str, sha: str, include: list[str], exclude: list[str]) -> list[tuple[str, str]]:
    listing = _git(repo, "ls-tree", "-r", "-z", "--full-tree", sha)
    files = []
    for entry in listing.split("\0"):
        if not entry or "\t" not in entry:
            continue
        meta, path = entry.split("\t", 1)
        parts = meta.split()
        if len(parts) < 3 or parts[1] != "blob":
            continue
        if any(fnmatch.fnmatch(path, g) for g in include) and not any(
            fnmatch.fnmatch(path, g) for g in exclude
        ):
            files.append((parts[2], path))
    return files


# ------------------------------------------------------------------------------- versions


def _version_tuple(label: str) -> tuple[int, ...] | None:
    numbers = re.findall(r"\d+", label)
    return tuple(int(n) for n in numbers) if numbers else None


def _below(label: str, floor: str | None) -> bool:
    mine, least = _version_tuple(label), _version_tuple(floor or "")
    return mine is not None and least is not None and mine < least


def _label(
    repo: str, tag: str, sha: str, rule: dict[str, Any], known: dict[str, str]
) -> str | None:
    if sha in known:
        return known[sha]
    label: str | None = tag
    if rule.get("tag_pattern"):
        match = re.search(str(rule["tag_pattern"]), tag)
        label = match.group(1) if match and match.groups() else (match.group(0) if match else None)
    elif rule.get("file") and rule.get("pattern"):
        try:
            text = _git(repo, "show", f"{sha}:{rule['file']}")
        except UsageError:
            text = ""
        match = re.search(str(rule["pattern"]), text)
        label = match.group(1) if match else None
    if label is not None:
        known[sha] = label
    return label


def _tags(client: dict[str, Any]) -> list[tuple[str, str]]:
    """The client's release tags, newest first, each with the commit it points at."""

    listing = _git(
        client["repo"],
        "for-each-ref",
        "--sort=-creatordate",
        "--format=%(refname:strip=2)%00%(objectname)%00%(*objectname)",
        "refs/tags",
    )
    tags = []
    for line in listing.splitlines():
        tag, obj, peeled = (line.split("\0") + ["", ""])[:3]
        if fnmatch.fnmatch(tag, str(client.get("tags") or "")):
            tags.append((tag, peeled or obj))
    return tags


def find_release(client: dict[str, Any], version: str) -> dict[str, Any] | None:
    """The newest tag whose version label is *version*, however old: ``{label, ref, sha}``."""

    rule = dict(client.get("version") or {})
    labels: dict[str, str] = {}
    for tag, sha in _tags(client) if client.get("tags") else []:
        if _label(client["repo"], tag, sha, rule, labels) == version:
            return {"label": version, "ref": tag, "sha": sha}
    return None


def _refs(
    client: dict[str, Any], labels: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The unreleased branch, then the newest tag of each of the newest released versions.

    Also returns what was left out, so a result can say which versions it never looked at.
    """

    repo = client["repo"]
    refs: list[dict[str, Any]] = []
    branch = client.get("branch") or ("origin/HEAD" if _rev(repo, "origin/HEAD") else "HEAD")
    head = _rev(repo, branch)
    if head is None:
        raise UsageError(f"client {client['name']}: branch {branch!r} does not resolve in {repo}")
    refs.append({"label": "unreleased", "ref": branch, "sha": head})
    if not client.get("tags"):
        return refs, None
    seen: set[str] = set()
    limit = int(client.get("max_versions") or _DEFAULT_MAX_VERSIONS)
    rule = dict(client.get("version") or {})
    tags = _tags(client)
    below = not_examined = 0
    for index, (tag, sha) in enumerate(tags):
        label = _label(repo, tag, sha, rule, labels)
        if label is None or label in seen:
            continue
        if _below(label, client.get("min_version")):
            below += 1
            continue
        seen.add(label)
        refs.append({"label": label, "ref": tag, "sha": sha})
        if len(seen) >= limit:
            not_examined = len(tags) - index - 1
            break
    history = {
        "max_versions": limit,
        "min_version": client.get("min_version"),
        "tags_not_examined": not_examined,
        "tags_below_min_version": below,
    }
    return refs, history


# -------------------------------------------------------------------------------- scanning


def _signature(client: dict[str, Any]) -> str:
    """Which scan results a cache holds: change the pattern and the cache starts over."""

    keyed = {k: client.get(k) for k in ("repo", "pattern", "base_path", "method", "version")}
    return hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()[:16]


def _scan_text(
    text: str, patterns: list[re.Pattern[str]], client: dict[str, Any]
) -> list[list[Any]]:
    calls = []
    default_method = str(client.get("method") or "*").upper()
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            method = (groups.get("method") or default_method).upper()
            path = normalize_path(groups["path"], str(client.get("base_path") or ""))
            calls.append([method, path, text.count("\n", 0, match.start()) + 1])
    return sorted(calls, key=lambda call: call[2])


def scan_client(
    client: dict[str, Any],
    cache_dir: Path,
    *,
    fetch: bool,
    extra_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Every call the client makes at each of its refs: ``{label: [call, ...]}``.

    ``extra_refs`` (``{label, ref, sha}``) adds versions outside the newest-tags window.
    """

    repo = client["repo"]
    store = cache_dir / f"client-{_signature(client)}.json"
    try:
        cache = json.loads(store.read_text())
    except (OSError, ValueError):
        cache = {}
    blobs: dict[str, list[list[Any]]] = cache.get("blobs") or {}
    labels: dict[str, str] = cache.get("labels") or {}
    fetched = _fetch(repo) if fetch else "skipped"
    refs, history = _refs(client, labels)
    known = {ref["label"] for ref in refs}
    refs += [dict(ref) for ref in extra_refs or [] if ref["label"] not in known]
    patterns = [re.compile(str(p), re.MULTILINE) for p in client["pattern"]]
    include = list(client["files"])
    exclude = list(client.get("exclude") or [])
    listed = {ref["label"]: _files_at(repo, ref["sha"], include, exclude) for ref in refs}
    unseen = sorted({blob for files in listed.values() for blob, _ in files if blob not in blobs})
    _prefetch(repo, unseen)
    for blob, text in _read_blobs(repo, unseen).items():
        blobs[blob] = _scan_text(text, patterns, client)
    # Not cached, so the next run tries again.
    unreadable = sum(blob not in blobs for blob in unseen)
    calls: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        calls[ref["label"]] = [
            {"method": m, "path": p, "file": path, "line": line}
            for blob, path in listed[ref["label"]]
            for m, p, line in blobs.get(blob, [])
        ]
        ref["calls"] = len(calls[ref["label"]])
    store.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(store, json.dumps({"blobs": blobs, "labels": labels}))
    return {
        "name": client["name"],
        "repo": repo,
        "fetch": fetched,
        "files_scanned_now": len(unseen) - unreadable,
        "files_unreadable": unreadable,
        "history": history,
        "refs": refs,
        "calls": calls,
    }


# -------------------------------------------------------------------------------- backend


def _run_spec_command(backend: dict[str, Any], checkout: str) -> dict[str, Any]:
    command = backend.get("spec_command")
    if not command:
        raise UsageError(
            "no `backend.spec_command` to export the OpenAPI spec",
            hint="Set it (with `{out}` for the file to write), or pass --base-spec/--head-spec.",
        )
    with tempfile.TemporaryDirectory(prefix="aua-spec-") as tmp:
        out = Path(tmp) / "openapi.json"
        rendered = str(command).format(
            out=shlex.quote(str(out)),
            repo=shlex.quote(backend["repo"]),
            checkout=shlex.quote(checkout),
        )
        proc = subprocess.run(
            rendered, shell=True, cwd=checkout, capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0 or not out.is_file():
            raise UsageError(
                f"spec_command failed in {checkout}: {(proc.stderr or proc.stdout).strip()[-800:]}",
                hint="Run it by hand in that folder; `{out}` must receive the OpenAPI JSON.",
            )
        return json.loads(out.read_text())


def _spec_at(
    backend: dict[str, Any], ref: str, cache_dir: Path, *, refresh: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = backend["repo"]
    sha = _rev(repo, ref)
    if sha is None:
        raise UsageError(f"backend ref {ref!r} does not resolve in {repo}")
    # A commit alone does not name an export: a different exporter can describe the same code
    # differently. `refresh` covers what the key cannot see, such as reinstalled dependencies.
    exporter = hashlib.sha256(str(backend.get("spec_command")).encode()).hexdigest()[:12]
    cached = cache_dir / "specs" / f"{sha}-{exporter}.json"
    info = {"ref": ref, "sha": sha}
    if cached.is_file() and not refresh:
        return json.loads(cached.read_text()), info
    # An export of a ref other than the working tree runs in an extracted copy, so the
    # backend checkout — and whatever someone has in progress there — is never touched.
    with tempfile.TemporaryDirectory(prefix="aua-backend-") as checkout:
        archive = subprocess.run(
            ["git", "-C", repo, "archive", "--format=tar", sha],
            capture_output=True,
            check=True,
            timeout=_GIT_TIMEOUT,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(checkout, filter="data")
            else:  # Python 3.11 before 3.11.4; the archive is the user's own repository
                tar.extractall(checkout)  # noqa: S202
        spec = _run_spec_command(backend, checkout)
    cached.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cached, json.dumps(spec))
    return spec, info


def _working_tree_spec(backend: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = backend["repo"]
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    return _run_spec_command(backend, repo), {
        "ref": "working tree",
        "sha": _rev(repo, "HEAD"),
        "dirty": dirty,
    }


def _load_spec_file(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    file = Path(path).expanduser()
    if not file.is_file():
        raise UsageError(f"no spec file at {file}")
    return json.loads(file.read_text()), {"ref": str(file)}


# ---------------------------------------------------------------------------------- report


def _who_calls(
    op_keys: list[str], clients: list[dict[str, Any]], routes: dict[str, dict[str, list[str]]]
) -> list[dict[str, Any]]:
    """Per client, every version that calls one of *op_keys*, each with its own sources."""

    users = []
    named = len(op_keys) > 1
    for client in clients:
        versions = []
        for ref in client["refs"]:
            hits = set()
            for call in client["calls"][ref["label"]]:
                reached = routes[client["name"]].get(f"{call['method']} {call['path']}", [])
                for op in op_keys:
                    if op in reached:
                        where = f"{call['file']}:{call['line']}"
                        hits.add(f"{where} ({op})" if named else where)
            if hits:
                versions.append(
                    {"version": ref["label"], "sha": ref["sha"], "sources": sorted(hits)}
                )
        if versions:
            users.append({"client": client["name"], "repo": client["repo"], "versions": versions})
    return users


def _group(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per change to one declaration: a shared model's field is changed once."""

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for change in changes:
        schema = change.get("schema")
        name = change.get("schema_field") if schema else change.get("field")
        scope = schema or change["operation"]
        key = (change["change"], scope, name, change.get("detail"))
        group = groups.setdefault(
            key,
            {
                "change": change["change"],
                "schema": schema,
                "field": name,
                "detail": change.get("detail"),
                "operations": [],
            },
        )
        group["operations"].append({"operation": change["operation"], "field": change.get("field")})
    return list(groups.values())


def _routes(
    client: dict[str, Any], known: dict[str, Any]
) -> tuple[dict[str, list[str]], list[str]]:
    """Each distinct client call → the spec operations it reaches, plus the calls reaching none."""

    routes: dict[str, list[str]] = {}
    for calls in client["calls"].values():
        for call in calls:
            key = f"{call['method']} {call['path']}"
            if key not in routes:
                routes[key] = match_operations(call["method"], call["path"], known)
    head = {f"{c['method']} {c['path']}" for c in client["calls"]["unreleased"]}
    return routes, sorted(k for k in head if not routes[k])


def _describe(group: dict[str, Any]) -> str:
    text = f"{group['schema'] or group['operations'][0]['operation']}: {group['change']}"
    if group.get("field"):
        text += f" `{group['field']}`"
    if group.get("detail"):
        text += f" ({group['detail']})"
    if len(group["operations"]) > 1:
        text += f" in {len(group['operations'])} operations"
    return text


def _scan_all(
    config: dict[str, Any], cache_dir: Path, *, fetch: bool, only: list[str]
) -> list[dict[str, Any]]:
    chosen = [c for c in config["clients"] if not only or c["name"] in only]
    if only and not chosen:
        raise UsageError(f"no client named {', '.join(only)} in {config['file']}")
    return [scan_client(c, cache_dir, fetch=fetch) for c in chosen]


def _client_summary(client: dict[str, Any], unmatched: list[str] | None = None) -> dict[str, Any]:
    summary = {
        "name": client["name"],
        "repo": client["repo"],
        "fetch": client["fetch"],
        "files_scanned_now": client["files_scanned_now"],
        "files_unreadable": client["files_unreadable"],
        "history": client["history"],
        "versions": [
            {k: ref[k] for k in ("label", "ref", "sha", "calls")} for ref in client["refs"]
        ],
    }
    if unmatched is not None:
        # Calls the base spec serves no route for: another service, a wrong `base_path`, or a
        # client already broken. A long list here usually means the config, not the apps.
        summary["unmatched_calls"] = {
            "count": len(unmatched),
            "sample": unmatched[:_UNMATCHED_SHOWN],
        }
    return summary


def check(
    config_path: str | Path,
    cache_dir: str | Path,
    *,
    base: str | None = None,
    head: str | None = None,
    base_spec: str | None = None,
    head_spec: str | None = None,
    fetch: bool = True,
    only: list[str] | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """What the backend change from *base* to *head* touches in every configured client."""

    config = load_config(config_path)
    backend = config["backend"]
    store = Path(cache_dir).expanduser() / "api-usage"
    if base_spec:
        spec_before, base_info = _load_spec_file(base_spec)
    else:
        base_ref = base or backend.get("base") or "origin/HEAD"
        spec_before, base_info = _spec_at(backend, base_ref, store, refresh=refresh)
    if head_spec:
        spec_after, head_info = _load_spec_file(head_spec)
    elif head:
        spec_after, head_info = _spec_at(backend, head, store, refresh=refresh)
    else:
        spec_after, head_info = _working_tree_spec(backend)
    diff = diff_specs(spec_before, spec_after)
    clients = _scan_all(config, store, fetch=fetch, only=only or [])
    known = operations(spec_before)
    routes: dict[str, dict[str, list[str]]] = {}
    summaries = []
    for client in clients:
        routes[client["name"]], unmatched = _routes(client, known)
        summaries.append(_client_summary(client, unmatched))
    findings, unused = [], []
    for group in _group(diff["changes"]):
        users = _who_calls([o["operation"] for o in group["operations"]], clients, routes)
        if users:
            findings.append({**group, "used_by": users})
        else:
            unused.append(_describe(group))
    findings.sort(
        key=lambda f: (
            f["change"] != "operation_removed",
            -sum(len(u["versions"]) for u in f["used_by"]),
        )
    )
    called = sorted({op for r in routes.values() for ops in r.values() for op in ops})
    untyped = [op for op in called if untyped_response(spec_before, known[op]["op"])]
    commits = []
    if base_info.get("sha") and head_info.get("sha") and base_info["sha"] != head_info["sha"]:
        log = _git(
            backend["repo"],
            "log",
            "--oneline",
            "--no-decorate",
            "-n",
            "30",
            f"{base_info['sha']}..{head_info['sha']}",
        )
        commits = log.splitlines()
    return {
        "ok": True,
        "summary": (
            f"{len(diff['changes'])} client-visible backend change(s) in "
            f"{len(findings) + len(unused)} group(s): {len(findings)} on operations a scanned "
            f"client version calls, {len(unused)} on operations none of them calls."
        ),
        "coverage": {
            "operations": {"base": len(known), "called": len(called)},
            "untyped_called_operations": {"count": len(untyped), "operations": untyped},
            "not_checked": NOT_CHECKED,
        },
        "findings": findings,
        "unused_changes": unused,
        "clients": summaries,
        "backend": {
            "base": base_info,
            "head": head_info,
            "operations": diff["operations"],
            "added_operations": len(diff["added_operations"]),
            "commits": commits,
        },
        "agent_brief": AGENT_BRIEF,
    }


def usage(
    config_path: str | Path,
    cache_dir: str | Path,
    *,
    endpoint: str | None = None,
    fetch: bool = True,
    only: list[str] | None = None,
) -> dict[str, Any]:
    """Every call each client makes, with the versions that make it; filter by *endpoint*."""

    config = load_config(config_path)
    store = Path(cache_dir).expanduser() / "api-usage"
    clients = _scan_all(config, store, fetch=fetch, only=only or [])
    needle = (endpoint or "").strip()
    table: dict[str, dict[str, Any]] = {}
    for client in clients:
        for ref in client["refs"]:
            for call in client["calls"][ref["label"]]:
                key = operation_key(call["method"], call["path"])
                if needle and needle.lower() not in key.lower():
                    continue
                entry = table.setdefault(key, {}).setdefault(
                    client["name"], {"versions": [], "sources": []}
                )
                if ref["label"] not in entry["versions"]:
                    entry["versions"].append(ref["label"])
                if ref["label"] == entry["versions"][0]:
                    entry["sources"].append(f"{call['file']}:{call['line']}")
    return {
        "ok": True,
        "clients": [_client_summary(c) for c in clients],
        "calls": [{"call": key, "clients": table[key]} for key in sorted(table)],
    }
