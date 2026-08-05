"""Per-device leases, so parallel agents stop fighting over the same emulator.

Three emulators attached, Claude testing search, Cursor testing delete — and both land on
``emulator-5554``, because ``connect(serial=None)`` takes "the only/first device". Each
agent then drives a screen the other is mutating. Nothing errors; the results are just
quietly wrong.

The design keeps two properties that matter more than features:

**No deadlock is possible.** Expiry is computed when a lease is *read*, not by a reaper
process. A stale lease is simply not a lease. There is no state a crashed agent can leave
behind that blocks anyone, and no cleanup step anyone can forget — which is exactly the
failure a "watchdog that frees stuck locks" design invites, because the watchdog becomes one
more thing that can die holding the world.

**Identity has to be stable across an agent's calls, or stickiness inverts into churn.**
Measured: a session id is *not* stable — consecutive tool calls from one agent reported sids
40966 then 40979, because each shell invocation gets its own session. Keying on that would
hand the agent a different emulator every command. Walking up to the first non-shell ancestor
finds the agent process itself (``claude``, ``cursor``, …), which lives for the whole run.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# Long enough that a single blocking call cannot outlive it: `--until` waits legitimately run
# 90-120s on a slow backend, and a lease that expires mid-wait would let another agent seize a
# device that is actively in use — strictly worse than no lease at all. Every command renews,
# so the TTL only ever measures the gap *since an agent stopped working*.
DEFAULT_TTL_S = 900

_SHELLS = {"sh", "bash", "zsh", "dash", "fish", "ksh", "csh", "tcsh", "login", "-zsh", "-bash"}


# --------------------------------------------------------------------------- identity


def _proc_name(pid: int) -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "comm=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return ""
    return (out.stdout or "").strip().rsplit("/", 1)[-1]


def _proc_ppid(pid: int) -> int | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        raw = (out.stdout or "").strip()
        return int(raw) if raw.isdigit() else None
    except Exception:
        return None


def _proc_started(pid: int) -> str:
    """Process start time — pairs with the pid so reuse cannot alias two agents."""
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return "".join((out.stdout or "").split())[-8:]
    except Exception:
        return ""


def derive_identity() -> str:
    """The first non-shell ancestor: the agent process, stable for its whole run.

    Falls back to this process when the walk finds nothing better, which is the right answer
    for a human at a terminal running one command.
    """
    pid: int | None = os.getpid()
    seen = 0
    while pid and pid > 1 and seen < 12:
        seen += 1
        parent = _proc_ppid(pid)
        if parent is None or parent <= 1:
            break
        name = _proc_name(parent)
        if name and name not in _SHELLS and not name.startswith("python"):
            return f"{name}-{parent}-{_proc_started(parent)}".strip("-")
        pid = parent
    return f"pid-{os.getpid()}-{_proc_started(os.getpid())}".strip("-")


def resolve_owner(explicit: str | None = None) -> str:
    """``--owner`` → ``$AUA_OWNER`` → derived. Never None: every caller is somebody."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    env = (os.environ.get("AUA_OWNER") or "").strip()
    if env:
        return env
    return derive_identity()


# --------------------------------------------------------------------------- storage


def lease_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "leases"


def _lease_path(cache_dir: str | Path, serial: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in serial)
    return lease_dir(cache_dir) / f"{safe}.json"


def _now() -> float:
    return time.time()


def _expired(entry: dict[str, Any], *, now: float | None = None) -> bool:
    now = _now() if now is None else now
    ttl = float(entry.get("ttl_s") or DEFAULT_TTL_S)
    return (now - float(entry.get("last_activity") or 0)) > ttl


def read_lease(cache_dir: str | Path, serial: str) -> dict[str, Any] | None:
    """The live lease on *serial*, or None when free/expired.

    Expiry is evaluated here rather than swept elsewhere — that is what makes a crashed
    agent's lease self-healing and a permanent block impossible.
    """
    path = _lease_path(cache_dir, serial)
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict) or _expired(entry):
        return None
    return entry


def _write(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def acquire(
    cache_dir: str | Path,
    serial: str,
    *,
    owner: str,
    ttl_s: int = DEFAULT_TTL_S,
    needs: list[str] | None = None,
    app: str | None = None,
) -> bool:
    """Claim *serial* for *owner*. True when held afterwards (including a sticky re-claim)."""
    path = _lease_path(cache_dir, serial)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "serial": serial,
        "owner": owner,
        "acquired": _now(),
        "last_activity": _now(),
        "ttl_s": int(ttl_s),
        "pid": os.getpid(),
        "needs": list(needs or []),
        "app": app,
    }

    # Fast path: nobody has ever held it.
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        pass
    except OSError:
        return True  # cannot coordinate on this filesystem; never block the caller
    else:
        with contextlib.suppress(Exception):
            os.write(fd, (json.dumps(entry, indent=2) + "\n").encode())
        with contextlib.suppress(Exception):
            os.close(fd)
        return True

    current = read_lease(cache_dir, serial)
    if current is None:
        # Free or expired — take it over, then re-read to settle a race with another
        # taker. Optimistic on purpose: the loser simply moves to another device.
        entry["acquired"] = _now()
        _write(path, entry)
        confirmed = read_lease(cache_dir, serial)
        return bool(confirmed and confirmed.get("owner") == owner)
    if current.get("owner") == owner:
        current["last_activity"] = _now()
        current["ttl_s"] = int(ttl_s)
        if app:
            current["app"] = app
        if needs:
            current["needs"] = list(needs)
        _write(path, current)
        return True
    return False


def renew(cache_dir: str | Path, serial: str, *, owner: str, app: str | None = None) -> bool:
    """Heartbeat. Called on every command, and from inside long waits."""
    current = read_lease(cache_dir, serial)
    if current is None or current.get("owner") != owner:
        return False
    current["last_activity"] = _now()
    if app:
        current["app"] = app
    _write(_lease_path(cache_dir, serial), current)
    return True


def release(cache_dir: str | Path, serial: str, *, owner: str | None = None) -> bool:
    """Drop the lease. A mismatched owner is refused so one agent cannot free another's."""
    current = read_lease(cache_dir, serial)
    if current is not None and owner is not None and current.get("owner") != owner:
        return False
    with contextlib.suppress(OSError):
        _lease_path(cache_dir, serial).unlink()
    return True


def list_leases(cache_dir: str | Path) -> list[dict[str, Any]]:
    """Live leases only — expired entries are reported as free, never as holders."""
    out: list[dict[str, Any]] = []
    for path in sorted(lease_dir(cache_dir).glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(entry, dict) and not _expired(entry):
            out.append(entry)
    return out


def holder(cache_dir: str | Path, serial: str) -> str | None:
    entry = read_lease(cache_dir, serial)
    return str(entry.get("owner")) if entry else None


def held_by(cache_dir: str | Path, owner: str) -> list[str]:
    """Serials this owner already holds — the basis for sticky assignment."""
    return [str(e["serial"]) for e in list_leases(cache_dir) if e.get("owner") == owner]


def idle_seconds(entry: dict[str, Any]) -> float:
    return max(0.0, _now() - float(entry.get("last_activity") or 0))


# --------------------------------------------------------------------------- capabilities

# Capabilities are probed from the *device*, not from an AVD's config.ini: that works for
# physical devices too, needs no serial→AVD mapping, and reports what is actually true rather
# than what was configured. Probing costs a few adb round-trips, so it is cached — and only
# runs at all when a caller passes --needs.
_CAPS_TTL_S = 3600


def _adb_shell(serial: str, command: str) -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["adb", "-s", serial, "shell", command],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


def probe_capabilities(cache_dir: str | Path, serial: str) -> dict[str, Any]:
    """What *serial* can do: ``root``, ``play``, ``proxy``.

    ``proxy`` means "an HTTPS-intercepting proxy can work here", which requires installing a
    CA into the system trust store — so it tracks rootability rather than being independent.
    """
    path = Path(cache_dir).expanduser() / "caps" / f"{serial.replace(':', '_')}.json"
    with contextlib.suppress(Exception):
        cached = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and (_now() - float(cached.get("probed") or 0)) < _CAPS_TTL_S:
            return cached

    tags = _adb_shell(serial, "getprop ro.build.tags")
    debuggable = _adb_shell(serial, "getprop ro.debuggable")
    vending = _adb_shell(serial, "pm list packages com.android.vending")
    rootable = ("test-keys" in tags) or debuggable.strip() == "1"
    caps: dict[str, Any] = {
        "serial": serial,
        "root": rootable,
        "play": "com.android.vending" in vending,
        "proxy": rootable,  # system-CA install is the gate
        "probed": _now(),
    }
    with contextlib.suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(caps, indent=2) + "\n", encoding="utf-8")
    return caps


# --------------------------------------------------------------------------- selection


def unmet_needs(capabilities: dict[str, Any] | None, needs: list[str] | None) -> list[str]:
    """Which of *needs* this device cannot satisfy.

    Unknown capability names are reported as unmet rather than ignored: silently satisfying
    ``--needs jellybean`` would hand back a device that does not do what was asked, and the
    caller would only find out several steps later.
    """
    if not needs:
        return []
    caps = capabilities or {}
    return [n for n in needs if not bool(caps.get(n))]


def choose_device(
    cache_dir: str | Path,
    *,
    owner: str,
    explicit: str | None,
    candidates: list[tuple[str, dict[str, Any]]],
    needs: list[str] | None = None,
    ttl_s: int = DEFAULT_TTL_S,
) -> tuple[str, str]:
    """Pick and claim a device. Returns ``(serial, why)``; raises when nothing is available.

    ``candidates`` is ``[(serial, capabilities)]`` for *online* devices, in preference order.

    Explicit intent is never silently redirected — asking for a specific emulator and being
    moved to another one would quietly invalidate a test pinned to that device's state. So an
    explicit ``--serial`` that is held fails loudly, while an unspecified one is free to route.
    """
    from .errors import DeviceLeasedError

    known = dict(candidates)

    def _free_report() -> str:
        free = [s for s, caps in candidates if holder(cache_dir, s) in (None, owner)
                and not unmet_needs(caps, needs)]
        return ", ".join(free) if free else "none"

    if explicit:
        current = read_lease(cache_dir, explicit)
        if current is not None and current.get("owner") != owner:
            raise DeviceLeasedError(
                f"{explicit} is leased by {current.get('owner')} "
                f"(active {idle_seconds(current):.0f}s ago)",
                hint=f"free now: {_free_report()} — omit --serial to auto-pick",
            )
        missing = unmet_needs(known.get(explicit), needs)
        if missing:
            raise DeviceLeasedError(
                f"{explicit} does not satisfy: {', '.join(missing)}",
                hint=f"free and matching: {_free_report()}",
            )
        acquire(cache_dir, explicit, owner=owner, ttl_s=ttl_s, needs=needs)
        return explicit, "explicit"

    # Sticky: keep an agent on the device it already knows, so its element ids, app state and
    # learned screen map stay valid across calls.
    for serial in held_by(cache_dir, owner):
        if serial in known and not unmet_needs(known[serial], needs):
            renew(cache_dir, serial, owner=owner)
            return serial, "sticky"

    for serial, caps in candidates:
        if unmet_needs(caps, needs):
            continue
        if read_lease(cache_dir, serial) is not None:
            continue
        if acquire(cache_dir, serial, owner=owner, ttl_s=ttl_s, needs=needs):
            return serial, "assigned"

    busy = [
        f"{e['serial']} ({e['owner']}, active {idle_seconds(e):.0f}s ago)"
        for e in list_leases(cache_dir)
        if e.get("serial") in known
    ]
    detail = "; ".join(busy) if busy else "no attached device matches"
    raise DeviceLeasedError(
        f"no free device{' matching ' + ','.join(needs) if needs else ''}: {detail}",
        hint="wait, start another emulator (`aua emulator start`), or widen --needs",
    )
