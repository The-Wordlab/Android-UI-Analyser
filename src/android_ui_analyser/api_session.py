"""Named API sessions: one client install kept across `aua api run`s, with a log of every call.

`aua api run FLOW --session NAME` keeps what the run saved (the sign-in token, a chat id) and
the ids the install made (`{{install.NAME}}`), so the next run with that name is the same user
on the same device: it skips the sign-in (`once: true`) and goes on from where the last run
stopped. Each run is logged in full beside a copy of the flow, and of the map version it was
checked against, so a session can be zipped and handed to someone else to continue.

A session holds a live sign-in, so it lives in the user's home with owner-only permissions,
never in a repository, and is removed after `KEEP_DAYS` without use.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from . import api_cartridge
from .api_map import version_context
from .atomic import atomic_write_text
from .errors import UsageError

ROOT = Path("~/.android-ui-analyser/api-sessions")
KEEP_DAYS = 14
FORMAT = 1
_FILE = "session.json"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_STAMP = "%Y-%m-%dT%H:%M:%SZ"


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(at: datetime) -> str:
    return at.strftime(_STAMP)


def _root(root: Path | None) -> Path:
    return (root or ROOT).expanduser()


def _check_name(name: str) -> str:
    if not _NAME.match(name):
        raise UsageError(
            f"{name!r} is not a session name",
            hint="Use up to 64 letters, digits, `.`, `_` or `-`, e.g. `bug-123`.",
        )
    return name


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


@dataclass
class Session:
    dir: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["name"])

    @property
    def runs(self) -> list[Path]:
        folder = self.dir / "runs"
        return sorted(folder.glob("*.json")) if folder.is_dir() else []

    def write(self) -> None:
        _mkdir(self.dir)
        atomic_write_text(self.dir / _FILE, json.dumps(self.data, indent=2, ensure_ascii=False))
        (self.dir / _FILE).chmod(0o600)

    def summary(self) -> dict[str, Any]:
        last = datetime.strptime(self.data["last_used"], _STAMP).replace(tzinfo=UTC)
        return {
            "name": self.name,
            "base_url": self.data.get("base_url"),
            "created": self.data["created"],
            "last_used": self.data["last_used"],
            "removed_after": _stamp(last + timedelta(days=KEEP_DAYS)),
            "runs": len(self.runs),
            "kept": sorted(self.data["saved"]),
            "dir": str(self.dir),
        }


def _load(folder: Path) -> Session | None:
    try:
        data = json.loads((folder / _FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        return None
    return Session(folder, data)


def prune(root: Path | None = None) -> list[str]:
    """Remove every session nobody has used for `KEEP_DAYS`; returns their names."""
    folder = _root(root)
    cutoff = _stamp(_now() - timedelta(days=KEEP_DAYS))
    removed = []
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        session = _load(path)
        if session and str(session.data.get("last_used", "")) < cutoff:
            shutil.rmtree(path)
            removed.append(session.name)
    return removed


def get(name: str, root: Path | None = None) -> Session:
    session = _load(_root(root) / _check_name(name))
    if session is None:
        raise UsageError(f"no session called {name!r}", hint="See `aua api session list`.")
    return session


def open_session(name: str, root: Path | None = None) -> Session:
    """The session called *name*, or a new one that is written once its first run is."""
    prune(root)
    folder = _root(root) / _check_name(name)
    if (folder / _FILE).exists():
        return get(name, root)
    now = _stamp(_now())
    data = {
        "format": FORMAT,
        "name": name,
        "created": now,
        "last_used": now,
        "base_url": None,
        "saved": {},
        "install": {},
    }
    return Session(folder, data)


def _copy_flow(session: Session, flow: Path) -> Path:
    """Keep the flow, and the map version it is checked against, inside the session."""
    if flow.resolve().is_relative_to(session.dir.resolve()):
        return flow  # a flow the session already holds, e.g. after an import
    mapped = version_context(flow)
    if not mapped:
        dest = session.dir / "flows" / flow.name
    else:
        version = session.dir / "map" / str(mapped["client"] or "client") / str(mapped["version"])
        _mkdir(version / "flows")
        meta = yaml.safe_load((flow.parent.parent / "version.yaml").read_text(encoding="utf-8"))
        # The copy holds a few of the version's flows, which a freeze would report as edits.
        meta.pop("frozen", None)
        (version / "version.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        schema = flow.parent.parent / "schema.yaml"
        if schema.is_file():
            shutil.copyfile(schema, version / "schema.yaml")
        dest = version / "flows" / flow.name
    _mkdir(dest.parent)
    shutil.copyfile(flow, dest)
    for shared in api_cartridge.load(flow).get("_includes", []):  # its `flow:` steps' cartridges
        shutil.copyfile(shared, dest.parent / Path(shared).name)
    return dest


def run(name: str, flow: str | Path, *, root: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    """`aua api run` as one more run of session *name*: same user, same install, logged.

    The run goes to the session's backend unless ``base_url`` says otherwise, since its user
    exists there only.
    """
    session = open_session(name, root)
    kwargs["base_url"] = kwargs.get("base_url") or session.data["base_url"]
    exchanges: list[dict[str, Any]] = []
    started = _now()
    result = api_cartridge.run(
        flow,
        saved=session.data["saved"],
        install=session.data["install"],
        exchanges=exchanges,
        **kwargs,
    )
    session.data["base_url"] = session.data["base_url"] or result["base_url"]
    session.data["last_used"] = _stamp(_now())
    session.write()
    copy = _copy_flow(session, Path(result["cartridge"]))
    _mkdir(session.dir / "runs")
    log = session.dir / "runs" / f"{len(session.runs) + 1:03d}-{copy.stem}.json"
    entry: dict[str, Any] = {"at": _stamp(started), "flow": str(copy.relative_to(session.dir))}
    entry.update(result=result, calls=exchanges)
    log.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    result["session"] = {
        "name": session.name,
        "new": len(session.runs) == 1,
        "kept": sorted(session.data["saved"]),
        "log": str(log),
    }
    return result


def list_sessions(root: Path | None = None) -> dict[str, Any]:
    removed = prune(root)
    folder = _root(root)
    paths = sorted(folder.iterdir()) if folder.is_dir() else []
    sessions = [s.summary() for s in map(_load, paths) if s]
    out: dict[str, Any] = {"sessions": sessions, "keep_days": KEEP_DAYS}
    if removed:
        out["removed_unused"] = removed
    return out


def delete(name: str, root: Path | None = None) -> dict[str, Any]:
    session = get(name, root)
    shutil.rmtree(session.dir)
    return {"ok": True, "deleted": name}


def export(name: str, out: str | Path | None = None, root: Path | None = None) -> dict[str, Any]:
    """Zip the session — its sign-in, logs, flows and map — for someone else to continue."""
    session = get(name, root)
    session.data["last_used"] = _stamp(_now())
    session.write()
    target = Path(out or f"{name}.aua-session.zip").expanduser().resolve()
    files = sorted(p for p in session.dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(session.dir).as_posix())
    return {
        "ok": True,
        "zip": str(target),
        "files": len(files),
        "warning": (
            f"it holds a live sign-in to {session.data.get('base_url')}: send it only to "
            "people who may act as that user. They continue with `aua api session import`."
        ),
    }


def _member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise UsageError(f"the zip holds a path outside its session: {name!r}")
    return path


def import_zip(path: str | Path, root: Path | None = None) -> dict[str, Any]:
    """Unpack a session another `aua api session export` made, to continue it here."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise UsageError(f"no file at {source}")
    try:
        zf = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise UsageError(f"{source} is not a zip") from exc
    with zf:
        members = [(info, _member(info.filename)) for info in zf.infolist() if not info.is_dir()]
        try:
            data = json.loads(zf.read(_FILE))
        except (KeyError, ValueError) as exc:
            raise UsageError(f"{source} is not an exported AUA API session") from exc
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise UsageError(f"{source} is not an exported AUA API session")
        name = _check_name(str(data.get("name")))
        prune(root)
        folder = _root(root) / name
        if folder.exists():
            raise UsageError(
                f"a session called {name!r} is already here",
                hint=f"Delete it first with `aua api session delete {name}`.",
            )
        _mkdir(folder)
        for info, rel in members:
            dest = folder.joinpath(*rel.parts)
            _mkdir(dest.parent)
            dest.write_bytes(zf.read(info))
    session = get(name, root)
    session.data["last_used"] = _stamp(_now())
    session.write()
    flows = sorted(str(p) for p in folder.rglob("flows/*.yaml"))
    return {"ok": True, "session": session.summary(), "flows": flows}
