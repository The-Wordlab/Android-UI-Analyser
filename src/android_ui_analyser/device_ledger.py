"""Write-ahead undo ledger: every persistent device change is reversible by *another* process.

A lease ends the moment its owner dies (``leases._expired`` checks the owner pid before it
checks the TTL), and expiry is evaluated lazily — on read, by whoever reads next. Nothing runs
at the instant a lease lapses. So a mutation whose undo lives only in the mutating process is
lost exactly when it is needed: the agent is SIGKILLed, the daemon stops, the lease frees, and
the next agent inherits a device that is still time-travelled, still offline, still proxied
through a mitmdump nobody owns.

Measured 2026-08-19 on this host: two orphan ``mitmdump`` processes alive, zero ownership
records, three emulators leased by processes that no longer exist.

Three properties make this recoverable by a stranger:

**The record is written before the device is touched.** A crash between the write and the
device call leaves a redundant undo — harmless and idempotent. A crash the other way around
leaves an unrecoverable device, and no watchdog, however alive, can fix it.

**The record lives at a fixed cross-process path, not under ``cache.dir``.** Parallel agents are
told to keep separate caches so their mock rules cannot leak into one another, which also means
the port agent A wrote into its own cache is invisible to agent B — and B is the one that
inherits the emulator. Mirrors ``proxy_mock.proxy_state_dir`` for exactly that reason.

**The undo is data, not a closure.** ``{"op": "set_http_proxy", "args": {...}}`` executed
through the selected :class:`~.platforms.base.PlatformAdapter`, so a reaper in an unrelated
process — or on another platform — can replay it without importing the code that recorded it.

Adding a device mutation? Register it in :data:`MUTATION_CATALOGUE` and record an entry from the
code that performs it. ``tests/test_every_device_mutation_registers_an_undo.py`` fails until you
do, which is the point: the next such feature must not be able to leak silently.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text

logger = logging.getLogger(__name__)

LEDGER_VERSION = 1

# How long a record with no live lease and no dead-owner proof is left alone. Deliberately NOT
# derived from ``leases.DEFAULT_TTL_S``: that 900s exists so a legitimate 90-120s ``--until``
# wait cannot lose its device, which is a different question from "how long may a poisoned
# device stay poisoned". The two deadlines must not share a number.
DEFAULT_GRACE_S = 120.0


# --------------------------------------------------------------------------- storage


def ledger_dir() -> Path:
    """Serial-keyed undo records, at a fixed path every process can find.

    Not under ``cache.dir`` on purpose — see the module docstring.
    """
    d = Path.home() / ".cache/android-ui-analyser/device-state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ledger_path(serial: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(serial))
    return ledger_dir() / f"{safe}.json"


@dataclass(frozen=True)
class Entry:
    """One reversible mutation.

    *key* makes re-recording idempotent: the same mutation performed twice replaces its record
    rather than queueing two undos. Include the varying part (``reverse_port:49097``) so two
    genuinely different mutations of one kind both survive.
    """

    key: str
    kind: str
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    owner: str | None = None
    owner_pid: int | None = None
    owner_started: str | None = None
    instance_token: str | None = None
    cache_dir: str | None = None
    # Was a lease governing this device when the change was made? It decides which signal means
    # "the agent is done": a vanished lease, or a vanished process. See reapable().
    leased: bool = True
    recorded: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "op": self.op,
            "args": dict(self.args),
            "detail": self.detail,
            "owner": self.owner,
            "owner_pid": self.owner_pid,
            "owner_started": self.owner_started,
            "instance_token": self.instance_token,
            "cache_dir": self.cache_dir,
            "leased": self.leased,
            "recorded": self.recorded,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Entry | None:
        key = raw.get("key")
        op = raw.get("op")
        if not isinstance(key, str) or not isinstance(op, str):
            return None
        args = raw.get("args")
        return cls(
            key=key,
            kind=str(raw.get("kind") or op),
            op=op,
            args=dict(args) if isinstance(args, dict) else {},
            detail=str(raw.get("detail") or ""),
            owner=raw.get("owner") if isinstance(raw.get("owner"), str) else None,
            owner_pid=raw.get("owner_pid") if isinstance(raw.get("owner_pid"), int) else None,
            owner_started=(
                raw.get("owner_started") if isinstance(raw.get("owner_started"), str) else None
            ),
            instance_token=(
                raw.get("instance_token") if isinstance(raw.get("instance_token"), str) else None
            ),
            cache_dir=raw.get("cache_dir") if isinstance(raw.get("cache_dir"), str) else None,
            leased=bool(raw.get("leased", True)),
            recorded=float(raw.get("recorded") or 0.0),
        )


def read_ledger(serial: str) -> list[Entry]:
    """Pending undos for *serial*, oldest first. Never raises: a corrupt file reads as empty."""
    path = ledger_path(serial)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(doc, dict):
        return []
    raw = doc.get("entries")
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict) and (entry := Entry.from_json(item)) is not None:
            out.append(entry)
    return out


def _write_ledger(serial: str, entries: list[Entry]) -> None:
    path = ledger_path(serial)
    if not entries:
        with contextlib.suppress(OSError):
            path.unlink()
        return
    payload = {
        "version": LEDGER_VERSION,
        "serial": str(serial),
        "written": time.time(),
        "entries": [e.to_json() for e in entries],
    }
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def record(
    serial: str,
    *,
    key: str,
    kind: str,
    op: str,
    args: dict[str, Any] | None = None,
    detail: str = "",
    owner: str | None = None,
    owner_pid: int | None = None,
    owner_started: str | None = None,
    instance_token: str | None = None,
    cache_dir: str | Path | None = None,
    leased: bool = True,
) -> Entry:
    """Record an undo **before** performing the mutation. Idempotent on *key*.

    Callers must not swallow failures here silently and then mutate anyway: an unrecorded
    mutation is the one failure mode this module exists to prevent.
    """
    if op not in UNDO_OPS:
        raise ValueError(
            f"unknown undo op {op!r}; register it in device_ledger.UNDO_OPS "
            f"(known: {', '.join(sorted(UNDO_OPS))})"
        )
    entry = Entry(
        key=key,
        kind=kind,
        op=op,
        args=dict(args or {}),
        detail=detail,
        owner=str(owner) if owner else None,
        owner_pid=owner_pid,
        owner_started=owner_started,
        instance_token=instance_token,
        cache_dir=str(cache_dir) if cache_dir else None,
        leased=bool(leased),
        recorded=time.time(),
    )
    entries = [e for e in read_ledger(serial) if e.key != key]
    entries.append(entry)
    _write_ledger(serial, entries)
    return entry


def forget(serial: str, *keys: str) -> int:
    """Drop records by key — call after undoing a mutation deliberately (``proxy stop``)."""
    entries = read_ledger(serial)
    kept = [e for e in entries if e.key not in keys]
    if len(kept) != len(entries):
        _write_ledger(serial, kept)
    return len(entries) - len(kept)


def clear(serial: str) -> None:
    _write_ledger(serial, [])


def pending_serials() -> list[str]:
    """Serials with at least one pending undo."""
    out = []
    for path in sorted(ledger_dir().glob("*.json")):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and doc.get("entries") and isinstance(doc.get("serial"), str):
                out.append(str(doc["serial"]))
    return out


# --------------------------------------------------------------------------- liveness


def _proc_started(pid: int) -> str | None:
    """Process start time, or ``None`` when it could not be determined.

    ``None`` and ``""`` are different answers and conflating them is dangerous: an unreadable
    ``ps`` must never read as "that process is a different one now", because the caller uses
    that to decide whether it may undo a live agent's work.
    """
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return None
    text = "".join((out.stdout or "").split())
    return text[-8:] if text else ""


def owner_state(entry: Entry) -> str:
    """``"gone"``, ``"alive"``, or ``"unknown"`` for the process that made the mutation."""
    pid = entry.owner_pid
    if not isinstance(pid, int) or pid <= 1:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        pass  # exists, just not ours to signal
    except OSError:
        return "unknown"
    if not entry.owner_started:
        return "alive"
    started = _proc_started(pid)
    if started is None:
        return "unknown"  # could not read; never guess "recycled" and undo a live run
    return "alive" if started == entry.owner_started else "gone"


def _lease_dirs(entries: list[Entry], extra: str | Path | None) -> list[Path]:
    """The host-wide registry plus legacy cache dirs that may still hold old leases.

    Current AUA writes one shared registry even when workers isolate their run caches. Legacy
    entries are still consulted during migration; any live lease found anywhere means hands off.
    """
    seen: dict[str, Path] = {}
    candidates = [
        os.environ.get("AUA_LEASE__REGISTRY_DIR") or "~/.cache/android-ui-analyser",
        *(e.cache_dir for e in entries),
    ]
    candidates.append(str(extra) if extra else None)
    # The documented override before the built-in default: a user who moved their cache — or a
    # test suite that redirects it — must not have the reaper consult a lease store nobody
    # writes to, nor one it was never meant to see.
    candidates.append(os.environ.get("AUA_CACHE__DIR") or "~/.cache/android-ui-analyser")
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        seen.setdefault(str(path), path)
    return list(seen.values())


def reapable(
    serial: str,
    *,
    entries: list[Entry] | None = None,
    cache_dir: str | Path | None = None,
    lease_registry_dir: str | Path | None = None,
    grace_s: float = DEFAULT_GRACE_S,
    now: float | None = None,
) -> str | None:
    """Why *serial*'s pending undos may be replayed, or ``None`` to leave it alone.

    Conservative by construction: every returned reason is positive evidence that no live
    holder is using the mutation. Undoing a running agent's proxy or clock mid-flow is strictly
    worse than leaving a stale one behind — the first breaks a test that was working, the second
    is visible in ``aua teardown status``.
    """
    from . import leases

    entries = read_ledger(serial) if entries is None else entries
    if not entries:
        return None

    # ``cache_dir`` is retained as the legacy coordination argument. Current callers pass the
    # explicit host-wide registry so cleanup cannot mistake an isolated run cache for authority.
    lease_authority = lease_registry_dir if lease_registry_dir is not None else cache_dir
    for directory in _lease_dirs(entries, lease_authority):
        if leases.read_lease(directory, serial) is not None:
            return None  # someone holds it right now

    states = {owner_state(e) for e in entries}
    if states == {"gone"}:
        pids = sorted({e.owner_pid for e in entries if e.owner_pid})
        return f"its owner process ({', '.join(str(p) for p in pids)}) is gone"

    now = time.time() if now is None else now
    newest = max(e.recorded for e in entries)
    idle = now - newest

    # A lease was in play when the change was made, and there is none now: the agent handed the
    # device back, which is the user's "out of lease" and the whole point of the ledger. Waiting
    # for the *process* to die as well would mean a long-lived orchestrator that moved on to
    # another device kept the first one dirty for the rest of its life.
    if any(e.leased for e in entries):
        if idle >= grace_s:
            return f"its lease is gone and the last change is {idle:.0f}s old"
        return None

    # Leasing was off when the change was made, so there is no ownership signal but the process
    # itself. A live one is presumed still working — reaping here would break a running run, and
    # with leasing off there is nothing else that could tell us otherwise.
    if "alive" in states:
        return None
    if idle >= grace_s:
        return f"no lease, no live owner, and the last change is {idle:.0f}s old"
    return None


# --------------------------------------------------------------------------- undo ops


@dataclass
class UndoContext:
    """What an undo handler may touch. Platform access is via the adapter only."""

    serial: str
    device: Any | None = None
    capability: Callable[[str], Any] | None = None
    instance_token: str | None = None

    def require_device(self) -> Any:
        if self.device is None:
            raise RuntimeError("this undo needs a connected target")
        return self.device

    def require_capability(self, name: str) -> Any:
        if self.capability is None:
            raise RuntimeError(f"this undo needs the {name!r} capability")
        return self.capability(name)


def _undo_set_http_proxy(ctx: UndoContext, args: dict[str, Any]) -> str:
    host_port = args.get("host_port")
    ctx.require_device().set_http_proxy(host_port or None)
    return f"http_proxy → {host_port or 'cleared'}"


def _undo_remove_reverse_port(ctx: UndoContext, args: dict[str, Any]) -> str:
    port = int(args.get("port") or 0)
    if port <= 0:
        return "no port recorded"
    ctx.require_device().remove_reverse_port(port)
    return f"reverse tcp:{port} removed"


def _undo_kill_host_process(ctx: UndoContext, args: dict[str, Any]) -> str:
    """Stop a host-side helper process this agent spawned (mitmdump, …).

    Verifies the command line still matches before signalling. A bare pid read from a file is
    not evidence: pids are recycled, and ``stop_mitm`` SIGTERMing whatever now holds one is how
    an unrelated process gets killed on a busy host.
    """
    pid = int(args.get("pid") or 0)
    match = str(args.get("match") or "")
    if pid <= 1:
        return "no pid recorded"
    command = _command_of(pid)
    if command is None:
        return f"pid {pid} already gone"
    if match and match not in command:
        return f"pid {pid} is now {command.split()[0] if command else '?'} — left alone"
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if _command_of(pid) is None:
            return f"pid {pid} stopped"
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, signal.SIGKILL)
    return f"pid {pid} killed"


def _command_of(pid: int) -> str | None:
    """Full command line of *pid*, or ``None`` when no such process exists."""
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return None
    text = (out.stdout or "").strip()
    return text or None


def _undo_restore_network_controls(ctx: UndoContext, args: dict[str, Any]) -> str:
    network = ctx.require_capability("network")
    cache = Path(str(args.get("cache_dir") or "~/.cache/android-ui-analyser")).expanduser()
    backup = network.load_backup(network.backup_path(cache, ctx.serial))
    if backup is None:
        return "no saved network state"
    network.restore_controls(ctx.require_device(), backup.state)
    return "network controls restored"


def _undo_restore_network_profile(ctx: UndoContext, args: dict[str, Any]) -> str:
    """Reverse whichever kind of profile was applied — radios, link shaping, or packet loss.

    One op rather than three because the restore point itself records which was used, and a
    reaper reading the ledger must not have to guess: undoing ``slow`` with the radio restore
    would leave the emulator still shaped to EDGE speeds while reporting success.
    """
    profiles = ctx.require_capability("network_profiles")
    cache = Path(str(args.get("cache_dir") or "~/.cache/android-ui-analyser")).expanduser()
    backup = profiles.load_profile(profiles.profile_path(cache, ctx.serial))
    if backup is None:
        return "no saved network profile"
    if profiles.stale_profile(backup, ctx.require_device()):
        return f"restore point for {backup.profile} predates this boot — already gone"
    if backup.profile == "slow":
        if backup.emulator_shape is None:
            return f"restore point for {backup.profile} has no original shape"
        profiles.restore_emulator_shape(ctx.serial, backup.emulator_shape)
        return "emulator link shaping restored"
    if backup.profile in profiles.PROFILE_NAMES and backup.interface:
        profiles.remove_loss(ctx.serial, backup)
        return f"packet loss removed (was {backup.profile})"
    profiles.restore_radio_profile(
        ctx.require_device(),
        backup.network_state,
        timeout_ms=int(args.get("timeout_ms") or 20_000),
    )
    return f"radio profile restored (was {backup.profile})"


def _undo_restore_developer_settings(ctx: UndoContext, args: dict[str, Any]) -> str:
    devsettings = ctx.require_capability("developer_settings")
    path = Path(str(args.get("backup_path") or ""))
    if not path.is_file():
        return "no saved developer settings"
    devsettings.anim_restore(ctx.require_device().shell, path)
    return "animation scales restored"


def _undo_clear_proxy_ownership(ctx: UndoContext, args: dict[str, Any]) -> str:
    """Drop the "who owns this device's proxy" record once the proxy is gone.

    Needs no target, so it still runs for a device that has been unplugged. Leaving the record
    behind is not fatal — the next ``proxy start`` reads it, finds the pid dead and reaps it —
    but until then ``aua proxy status`` and any parallel agent are told a proxy is live that
    this very teardown just removed.
    """
    proxy = ctx.require_capability("proxy")
    proxy.clear_state(ctx.serial)
    return "proxy ownership record cleared"


def _undo_clear_mock_rules(ctx: UndoContext, args: dict[str, Any]) -> str:
    """Disarm mock mode and drop every rule this cache dir's mitmproxy addon still reloads.

    Needs no target: the rules file is a host-side JSON sidecar the addon hot-reloads from
    disk on every exchange, so a reaper clears it for an unplugged device exactly as well as
    for a live one — and must, because a left-armed stub or ``record`` mode silently poisons
    whichever agent inherits this cache dir next. Measured: 14 stale rules from unrelated
    earlier sessions, with ``mode: "map"`` left armed globally.
    """
    proxy = ctx.require_capability("proxy")
    cache_dir = str(args.get("cache_dir") or "~/.cache/android-ui-analyser")
    removed = proxy.clear_rules(cache_dir)
    return f"mock rules cleared ({removed} removed), mode disarmed"


def _undo_set_airplane_mode(ctx: UndoContext, args: dict[str, Any]) -> str:
    enabled = bool(args.get("enabled"))
    ctx.require_device().set_airplane_mode(enabled)
    return f"airplane mode → {'on' if enabled else 'off'}"


def _undo_set_clock(ctx: UndoContext, args: dict[str, Any]) -> str:
    previous = args.get("timestamp_ms")
    if not isinstance(previous, int):
        return "no saved clock"
    # A clock set N seconds ago must land back on *now*, not on the instant it was saved.
    drift_ms = int((time.time() - float(args.get("saved_at") or time.time())) * 1000)
    ctx.require_device().set_clock(previous + max(0, drift_ms))
    return "wall clock restored"


def _undo_restore_app_prefs(ctx: UndoContext, args: dict[str, Any]) -> str:
    """Put an app's own ``shared_prefs`` file back the way a setup step found it.

    Not cosmetic state: the whole point of writing prefs is to move a build onto a different
    backend or past its onboarding, so a record left unreplayed means the next agent picks up
    an app configured for someone else's test and has no way to see why.
    """
    flags = ctx.require_capability("feature_flags")
    path = Path(str(args.get("backup_path") or ""))
    if not path.is_file():
        return "no saved app preferences"
    return flags.restore_prefs(ctx.require_device(), path)


def _undo_disable_device_agent(ctx: UndoContext, args: dict[str, Any]) -> str:
    agent = ctx.require_capability("device_agent")
    agent.disable(ctx.serial)
    return "on-device helper service disabled"


@dataclass(frozen=True)
class UndoOp:
    """One replayable undo. *order* replays device-facing undos before host cleanup."""

    handler: Callable[[UndoContext, dict[str, Any]], str]
    needs_device: bool
    order: int
    summary: str


UNDO_OPS: dict[str, UndoOp] = {
    "set_http_proxy": UndoOp(_undo_set_http_proxy, True, 10, "un-point the device's HTTP proxy"),
    "remove_reverse_port": UndoOp(
        _undo_remove_reverse_port, True, 20, "drop the host port tunnel"
    ),
    "restore_network_controls": UndoOp(
        _undo_restore_network_controls, True, 30, "restore Wi-Fi / mobile data / airplane"
    ),
    "restore_network_profile": UndoOp(
        _undo_restore_network_profile, True, 31, "restore radio profile and link shaping"
    ),
    "set_airplane_mode": UndoOp(_undo_set_airplane_mode, True, 32, "restore airplane mode"),
    "restore_developer_settings": UndoOp(
        _undo_restore_developer_settings, True, 40, "restore animation scales"
    ),
    "restore_app_prefs": UndoOp(
        _undo_restore_app_prefs, True, 45, "restore the app's own shared preferences"
    ),
    "set_clock": UndoOp(_undo_set_clock, True, 50, "restore the wall clock"),
    "disable_device_agent": UndoOp(
        _undo_disable_device_agent, True, 60, "disable the on-device helper service"
    ),
    "kill_host_process": UndoOp(
        _undo_kill_host_process, False, 90, "stop the host helper process"
    ),
    # Last: the record is what tells a peer the proxy is live, so it must outlive the undos that
    # dismantle it. Clearing it first would open a window where another agent sees a free device
    # while its proxy is still half up.
    "clear_proxy_ownership": UndoOp(
        _undo_clear_proxy_ownership, False, 95, "clear the proxy ownership record"
    ),
    "clear_mock_rules": UndoOp(
        _undo_clear_mock_rules, False, 90, "clear armed mock rules and disarm mode"
    ),
}


# --------------------------------------------------------------------------- the catalogue
#
# Every persistent device mutation AUA can perform, and how it is undone. This is the list the
# architecture guard checks: a new mutation that is not here fails the suite rather than
# shipping a device the next agent inherits dirty.


@dataclass(frozen=True)
class Mutation:
    """A persistent device change, and the undo op that reverses it."""

    kind: str
    site: str  # "<module>:<function>" that performs it
    undo_op: str | None  # None only for a documented irreversible/self-healing change
    note: str


MUTATION_CATALOGUE: dict[str, Mutation] = {
    "http_proxy": Mutation(
        "http_proxy",
        "device.py:set_http_proxy",
        "set_http_proxy",
        "settings put global http_proxy — the change that leaves every app 'Offline' when the "
        "host mitmdump behind it dies.",
    ),
    "reverse_port": Mutation(
        "reverse_port",
        "device.py:adb_reverse",
        "remove_reverse_port",
        "Host port exposed to the target; dies with the adb transport but the device-side "
        "setting pointing at it does not.",
    ),
    "airplane_mode": Mutation(
        "airplane_mode",
        "device.py:set_airplane_mode",
        "set_airplane_mode",
        "`aua airplane on` on its own; the verified-offline path records the whole network "
        "snapshot under network_controls instead.",
    ),
    "network_controls": Mutation(
        "network_controls",
        "network.py:apply_offline_controls",
        "restore_network_controls",
        "svc wifi/data disable for verified offline testing.",
    ),
    "radio_profile": Mutation(
        "radio_profile",
        "network_profiles.py:apply_radio_profile",
        "restore_network_profile",
        "Radio generation + emulator link shaping + tc loss.",
    ),
    "developer_settings": Mutation(
        "developer_settings",
        "devopts.py:_settings_put",
        "restore_developer_settings",
        "Animation scales and crash dialogs.",
    ),
    "wall_clock": Mutation(
        "wall_clock",
        "device.py:set_clock",
        "set_clock",
        "Time travel invalidates auth tokens; a device left in the past 401s every login.",
    ),
    "app_prefs": Mutation(
        "app_prefs",
        "flags.py:write_prefs",
        "restore_app_prefs",
        "A flow step writing the app's own shared_prefs — which backend a build talks to, "
        "whether onboarding counts as seen. `app clear` would reset it, but nothing runs "
        "`app clear` on the way out, so the next agent inherits a differently configured app "
        "and no way to tell that AUA is why.",
    ),
    "device_agent_service": Mutation(
        "device_agent_service",
        "device_agent.py:enable",
        "disable_device_agent",
        "Secure accessibility-services list; Android suppresses it while uiautomator2 holds "
        "UiAutomation, so a left-enabled service is not inert.",
    ),
    "host_proxy_process": Mutation(
        "host_proxy_process",
        "proxy_mock.py:start_mitm",
        "kill_host_process",
        "mitmdump is spawned with start_new_session=True and outlives the agent that started "
        "it, still holding its listen port and still writing its cassette.",
    ),
    "proxy_ownership": Mutation(
        "proxy_ownership",
        "proxy_mock.py:write_state",
        "clear_proxy_ownership",
        "The cross-process record saying who owns this device's proxy. Host-side, but it speaks "
        "for the device, so it has to be retracted with the device change.",
    ),
    "mock_rules": Mutation(
        "mock_rules",
        "proxy_mock.py:write_rules",
        "clear_mock_rules",
        "The live HTTP mock/record rules sidecar the mitmproxy addon hot-reloads from disk. "
        "Host-side, but outlives the command that armed it: left in place, one session's stubs "
        "or record mode silently poison the next agent's traffic against the same cache dir.",
    ),
    # Mutations that need no undo. Listed rather than omitted so the guard can tell "considered
    # and safe" from "forgotten".
    "system_ca": Mutation(
        "system_ca",
        "proxy_mock.py:install_system_ca",
        None,
        "Written to a tmpfs overlay of the system trust store: gone on reboot, and an emulator "
        "for proxy testing is expected to trust the proxy CA.",
    ),
    "app_data": Mutation(
        "app_data",
        "engine.py:app",
        None,
        "`app clear` is the caller's explicit destructive intent, confirmed at the CLI; "
        "restoring it would undo what the agent asked for.",
    ),
    "app_database": Mutation(
        "app_database",
        "app_database.py:execute_database",
        None,
        "`db execute` already creates its own restore point and requires --yes; "
        "`aua db restore` is the documented rollback.",
    ),
}


def catalogue_gaps() -> list[str]:
    """Catalogue entries naming an undo op that does not exist. Used by the guard test."""
    return sorted(
        f"{m.kind} → unknown undo op {m.undo_op!r}"
        for m in MUTATION_CATALOGUE.values()
        if m.undo_op is not None and m.undo_op not in UNDO_OPS
    )


# --------------------------------------------------------------------------- replay


def replay(
    serial: str,
    *,
    entries: list[Entry] | None = None,
    context: UndoContext,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the pending undos for *serial*. Safe to call twice; each success is forgotten.

    A reboot since the record was written means the device already forgot the mutation, so the
    entry is dropped without touching the target — the same guard ``network_profiles`` applies
    to its own backups via ``instance_token``.
    """
    pending = read_ledger(serial) if entries is None else entries
    ordered = sorted(pending, key=lambda e: (UNDO_OPS[e.op].order if e.op in UNDO_OPS else 999))
    done: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for entry in ordered:
        op = UNDO_OPS.get(entry.op)
        if op is None:
            failed.append({"key": entry.key, "error": f"unknown undo op {entry.op!r}"})
            continue
        rebooted = (
            entry.instance_token is not None
            and context.instance_token is not None
            and entry.instance_token != context.instance_token
        )
        if rebooted and op.needs_device:
            done.append(
                {"key": entry.key, "kind": entry.kind, "result": "already gone (target rebooted)"}
            )
            if not dry_run:
                forget(serial, entry.key)
            continue
        if dry_run:
            done.append({"key": entry.key, "kind": entry.kind, "result": f"would {op.summary}"})
            continue
        try:
            detail = op.handler(context, entry.args)
        except Exception as exc:  # keep going: one stuck undo must not block the rest
            logger.warning("undo %s on %s failed: %s", entry.key, serial, exc)
            failed.append({"key": entry.key, "kind": entry.kind, "error": str(exc)})
            continue
        forget(serial, entry.key)
        done.append({"key": entry.key, "kind": entry.kind, "result": detail})
    return {
        "serial": serial,
        "undone": done,
        "failed": failed,
        "remaining": len(read_ledger(serial)),
    }


def status(
    *,
    cache_dir: str | Path | None = None,
    lease_registry_dir: str | Path | None = None,
    grace_s: float = DEFAULT_GRACE_S,
) -> list[dict[str, Any]]:
    """What is pending, per serial, and whether it may be reaped right now."""
    out = []
    for serial in pending_serials():
        entries = read_ledger(serial)
        why = reapable(
            serial,
            entries=entries,
            cache_dir=cache_dir,
            lease_registry_dir=lease_registry_dir,
            grace_s=grace_s,
        )
        out.append(
            {
                "serial": serial,
                "reapable": why is not None,
                "why": why or "a live holder still owns these changes",
                "changes": [
                    {
                        "kind": e.kind,
                        "key": e.key,
                        "detail": e.detail,
                        "owner": e.owner,
                        "owner_state": owner_state(e),
                        "age_s": round(max(0.0, time.time() - e.recorded), 1),
                    }
                    for e in entries
                ],
            }
        )
    return out


__all__ = [
    "DEFAULT_GRACE_S",
    "Entry",
    "MUTATION_CATALOGUE",
    "Mutation",
    "UNDO_OPS",
    "UndoContext",
    "UndoOp",
    "catalogue_gaps",
    "clear",
    "forget",
    "ledger_dir",
    "ledger_path",
    "owner_state",
    "pending_serials",
    "read_ledger",
    "reapable",
    "record",
    "replay",
    "status",
]
