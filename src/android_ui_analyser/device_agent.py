"""Android-only capability service for the optional on-device helper APK.

The helper is an AccessibilityService that pushes screen-change events and reads the view
hierarchy in-process. It exists because AUA's default path has to *guess* when a screen
settled: it polls screenshots until three agree. Measured on this project's own devices, that
poll costs 183-242ms on an emulator and 490ms on a physical phone, per action, just to decide
whether anything happened. The helper is told instead, at 75ms.

Nothing here is on by default. ``helper.enabled`` is false, every entry point refuses cleanly
when the helper is absent, and callers keep their polling path. The helper is strictly a
faster answer to a question AUA already knows how to ask.

Reached only through ``AndroidPlatform.capability("device_agent")`` — never imported by the
engine, CLI, MCP or daemon. Direct adb use is correct *here*; that is what an Android-only
capability module is for.
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .errors import DeviceError

PACKAGE = "dev.aua.helper"
SERVICE = f"{PACKAGE}/{PACKAGE}.HelperService"
# Must match android:label on the service in the helper manifest: `dumpsys accessibility`
# reports bound services by label only, so this string is how the host recognises its own.
LABEL = "AUA Helper"
DEVICE_PORT = 8779

# Bump together with InfoFeature.PROTOCOL in the APK. A device carrying an older helper is
# refused rather than driven through a wire format it does not speak.
PROTOCOL = 1

_SECURE_SERVICES = "secure enabled_accessibility_services"
_SECURE_ENABLED = "secure accessibility_enabled"


class HelperUnavailableError(DeviceError):
    """The helper cannot be used on this target, with the reason attached."""


def apk_path() -> Path:
    """Filesystem path of the bundled helper APK."""

    return Path(str(resources.files("android_ui_analyser") / "data" / "aua-helper.apk"))


def _adb(serial: str, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["adb", "-s", serial, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _shell(serial: str, command: str, *, timeout: float = 30.0) -> str:
    return (_adb(serial, "shell", command, timeout=timeout).stdout or "").strip()


def _setting(serial: str, key: str) -> str:
    raw = _shell(serial, f"settings get {key}")
    return "" if raw in {"", "null"} else raw


# --------------------------------------------------------------------------- root


def rootable(serial: str) -> bool:
    """Cheap, read-only guess at whether ``adb root`` could work — no side effects.

    ``root_available`` answers the same question by *running* ``adb root``, which restarts
    adbd and costs about a second. That is fine once during setup and completely wrong as a
    per-run check: with the helper simply switched on, every flow would have paid it on a
    device that was never going to allow root. A userdebug build advertises itself, so read
    the properties instead and only escalate when they look promising.
    """

    if _shell(serial, "id -u") == "0":
        return True
    tags = _shell(serial, "getprop ro.build.tags")
    debuggable = _shell(serial, "getprop ro.debuggable")
    return "test-keys" in tags or debuggable == "1"


def root_available(serial: str) -> bool:
    """Can adbd run as root here? Google Play images and retail phones cannot."""

    if _shell(serial, "id -u") == "0":
        return True
    result = _adb(serial, "root", timeout=30)
    blob = ((result.stdout or "") + (result.stderr or "")).lower()
    if any(word in blob for word in ("cannot", "production", "unauthorized", "not allowed")):
        return False
    _adb(serial, "wait-for-device", timeout=60)
    time.sleep(0.5)
    return _shell(serial, "id -u") == "0"


# --------------------------------------------------------------------------- lifecycle


def release_uiautomation(serial: str) -> None:
    """Stop the on-device uiautomator2 server so accessibility services can bind again.

    Scoped to one serial by construction — every call goes through ``adb -s <serial>`` — so a
    parallel agent driving another device is unaffected. Within a device this is safe because
    AUA's leases give one agent the target at a time, and the Python client reconnects on its
    next call.

    Killing rather than asking politely: the server outlives the client that started it (it is
    an ``app_process``, not a child), so it survives the CLI exiting and a daemon stop, and it
    goes on holding UiAutomation — and therefore keeps the helper suppressed — indefinitely.

    Returns without waiting. Waiting here used to be a flat one-second sleep, which was a
    third of the entire handover and bought nothing: the caller asks :func:`is_bound` next,
    and that already polls until the framework has actually rebound — measured at 21ms once
    the slot is free. Sleeping first only delayed the same answer.
    """

    _shell(serial, "pkill -f com.wetest.uia2.Main", timeout=15)


def uiautomation_held(serial: str) -> bool:
    """Is uiautomator2's on-device server running, and therefore holding UiAutomation?

    While it is, Android suppresses every accessibility service, so the helper is not bound no
    matter how healthy it is. Reporting a bare "not bound" in that state reads as a broken
    helper when the real answer is "asleep, by design, because AUA itself is driving".
    """

    return bool(_shell(serial, "pidof -s app_process", timeout=15).strip()) and bool(
        _shell(serial, "ps -A -o ARGS", timeout=20).find("com.wetest.uia2.Main") >= 0
    )


def is_installed(serial: str) -> bool:
    return bool(_shell(serial, f"pm list packages {PACKAGE}").strip())


def is_enabled(serial: str) -> bool:
    return SERVICE in _setting(serial, _SECURE_SERVICES)


def is_bound(serial: str, *, settle_s: float = 3.0) -> bool:
    """Enabled is not running.

    Android will happily record a sideloaded accessibility service as *enabled* and then
    never bind it, so ``enabled`` is not a readiness signal.

    Two traps here, both hit in practice. ``dumpsys accessibility`` identifies a bound service
    by its **label**, never by package, so matching the package against that dump silently
    never matches. And process liveness is not a substitute: the helper's channel threads keep
    the process alive across a rebind, so ``pidof`` reports a live process while no service
    instance is attached.

    The label match against the bound set is therefore the signal, and it is the framework's
    own answer rather than an inference.

    *settle_s* exists because AUA disturbs this itself: connecting uiautomator2 takes a
    UiAutomation connection, and the framework responds by tearing down and re-creating every
    accessibility service. Any check right after an AUA command can therefore land in a
    sub-second window where nothing is bound, and answering "no" there would be a false
    negative about a perfectly healthy helper.
    """

    deadline = time.monotonic() + max(0.0, settle_s)
    while True:
        dump = _shell(serial, "dumpsys accessibility", timeout=20)
        for line in dump.splitlines():
            if "Bound services" in line:
                if f"label={LABEL}" in line:
                    return True
                break
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.4)


def install(serial: str, *, reinstall: bool = False, force: bool = False) -> dict[str, Any]:
    """Push the helper APK, refusing a target that could never run it.

    An accessibility service Android will not bind is not a harmless spare app: it is 900KB
    of litter on someone's phone, listed under Accessibility, doing nothing. *force* exists
    for deliberate cases (staging an image that will be rooted later), not for routine use.
    """

    if not force and not rootable(serial):
        raise HelperUnavailableError(
            f"{serial} is not a rootable target, so the helper would be installed and never run",
            code="helper_needs_root",
            hint="Use a Google APIs emulator, or pass --force to install anyway.",
        )
    apk = apk_path()
    if not apk.exists():
        raise HelperUnavailableError(
            "the bundled helper APK is missing from this install",
            code="helper_apk_missing",
            hint="Rebuild it with `helper/tools/build.py`, which copies it into "
            "src/android_ui_analyser/data/aua-helper.apk.",
        )
    if is_installed(serial) and not reinstall:
        return {"installed": True, "action": "already-present"}
    result = _adb(serial, "install", "-r", "-g", str(apk), timeout=180)
    blob = (result.stdout or "") + (result.stderr or "")
    if "Success" not in blob:
        raise HelperUnavailableError(
            f"helper install failed on {serial}: {blob.strip()[:200]}",
            code="helper_install_failed",
        )
    return {"installed": True, "action": "installed"}


def enable(serial: str) -> dict[str, Any]:
    """Refuse unless the target can run it, then install, switch on, and confirm it bound.

    Order matters and used to be wrong: this installed first and checked root second, so a
    retail phone got a 900KB APK pushed onto it and *then* the refusal — leaving an app behind
    that can never run. Nothing should land on a device that cannot use it.
    """

    # Read-only probe first, so a device that was never going to qualify is not touched at all.
    if not rootable(serial):
        raise HelperUnavailableError(
            f"{serial} is not a rootable target, so the helper cannot run there",
            code="helper_needs_root",
            hint=(
                "Android will not bind a sideloaded accessibility service unless adbd can run "
                "as root, which rules out retail phones and Play-image AVDs. Use a Google APIs "
                "emulator (`aua emulator recommend-proxy`), or switch 'AUA Helper' on by hand "
                "under Settings > Accessibility. AUA keeps using its polling path until then."
            ),
        )

    if not root_available(serial):
        raise HelperUnavailableError(
            f"{serial} cannot run adbd as root, so the helper cannot be switched on for you",
            code="helper_needs_root",
            hint=(
                "Android refuses to bind a sideloaded accessibility service that only `adb "
                "shell settings` enabled. Use a rootable Google APIs emulator, or enable "
                "'AUA Helper' by hand under Settings > Accessibility (one time, it persists). "
                "AUA keeps using its polling path until then."
            ),
        )

    if not is_installed(serial):
        install(serial)

    # Android 13+ treats accessibility for a sideloaded app as a "restricted setting" and
    # will record the service as enabled while silently never binding it. Granting the appop
    # is what turns the switch from cosmetic into real; without it every call below appears
    # to succeed and the helper never starts.
    _shell(serial, f"cmd appops set {PACKAGE} ACCESS_RESTRICTED_SETTINGS allow")

    # Preserve any service already enabled here — clobbering the list would silently switch
    # off a screen reader someone depends on.
    current = [s for s in _setting(serial, _SECURE_SERVICES).split(":") if s and s != SERVICE]
    merged = ":".join([*current, SERVICE])

    # Android suppresses every accessibility service while a UiAutomation connection is held,
    # and every AUA command — including the one that runs this — connects uiautomator2. So the
    # helper could be switched on perfectly and still never bind while we watched for it, which
    # is exactly what made this look intermittent. Release the slot before waiting.
    release_uiautomation(serial)

    def _write_switch_on() -> None:
        _shell(serial, f"settings put {_SECURE_SERVICES} {merged}")
        _shell(serial, f"settings put {_SECURE_ENABLED} 1")

    def _bound_within(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if is_bound(serial, settle_s=0.0):
                return True
            time.sleep(0.5)
        return False

    # Two attempts on purpose. Straight after a fresh install the write can land while the
    # package is still being scanned: the setting sticks, and the service is simply never
    # picked up. Toggling once more after that window reliably takes, and is harmless when
    # the first attempt already worked.
    _write_switch_on()
    if not _bound_within(15.0):
        _shell(serial, f"settings put {_SECURE_ENABLED} 0")
        time.sleep(1.0)
        _write_switch_on()
        if not _bound_within(20.0):
            raise HelperUnavailableError(
                f"helper was enabled on {serial} but Android never bound it",
                code="helper_not_bound",
                hint=(
                    "Check `adb -s <serial> shell dumpsys accessibility` for a Crashed "
                    "services entry. AUA keeps working on its polling path regardless."
                ),
            )
    return {"enabled": True, "bound": True, "preserved": current}


def disable(serial: str) -> dict[str, Any]:
    """Remove only our entry, leaving any other accessibility service enabled."""

    remaining = [s for s in _setting(serial, _SECURE_SERVICES).split(":") if s and s != SERVICE]
    _shell(serial, f"settings put {_SECURE_SERVICES} {':'.join(remaining)}")
    if not remaining:
        _shell(serial, f"settings put {_SECURE_ENABLED} 0")
    return {"enabled": False, "remaining": remaining}


def remove(serial: str) -> dict[str, Any]:
    disable(serial)
    _adb(serial, "uninstall", PACKAGE, timeout=60)
    return {"installed": False, "enabled": False}


def status(serial: str) -> dict[str, Any]:
    """Report readiness, preferring the helper's own answer over an outside guess.

    A successful handshake that reports ``service_bound`` is ground truth. The dumpsys route
    is only a fallback for when no channel can be opened, because it races with every
    UiAutomation connect and reports false negatives on a healthy helper.
    """

    installed = is_installed(serial)
    enabled = installed and is_enabled(serial)
    bound = False
    version: str | None = None
    features: list[str] = []
    if enabled:
        try:
            channel = _connect_channel(serial, timeout=3.0)
        except HelperUnavailableError:
            bound = is_bound(serial)
        else:
            try:
                info = channel.request("helper.info", timeout=3.0)
                bound = bool(info.get("service_bound"))
                version = info.get("version")
                features = list(info.get("features") or [])
            except HelperUnavailableError:
                bound = is_bound(serial)
            finally:
                channel.close()
    suppressed = bool(enabled and not bound and uiautomation_held(serial))
    return {
        "package": PACKAGE,
        "apk_present": apk_path().exists(),
        "installed": installed,
        "enabled": enabled,
        "bound": bound,
        # Distinguishes "the helper is broken" from "AUA is holding the UiAutomation slot, so
        # the helper is asleep" — the same reading, two completely different situations.
        "suppressed_by_uiautomation": suppressed,
        "ready": bool(bound or suppressed),
        "version": version,
        "features": features,
        "protocol": PROTOCOL,
    }


# --------------------------------------------------------------------------- tree shape


# The helper reports a node with the names an accessibility API uses; the hierarchy
# normalizer reads the names uiautomator's XML dump uses. Only three differ.
_TREE_FIELD_TO_XML = {
    "text": "text",
    "class": "class",
    "package": "package",
    "rid": "resource-id",
    "desc": "content-desc",
    "bounds": "bounds",
    "clickable": "clickable",
    "long_clickable": "long-clickable",
    "checkable": "checkable",
    "checked": "checked",
    "enabled": "enabled",
    "focused": "focused",
    "focusable": "focusable",
    "scrollable": "scrollable",
    "selected": "selected",
    "password": "password",
    "visible": "displayed",
}


def tree_to_xml(tree: dict[str, Any]) -> str:
    """Render a ``ui.tree`` reply as the uiautomator XML the normalizer already parses.

    Deliberately a translation rather than a second normalizer. Element semantics — what
    counts as interesting, how a label is chosen, which nodes are system chrome — are subtle
    and already settled in one place; growing a parallel implementation for the helper would
    mean two answers to the same question, drifting apart quietly.

    **Not a substitute for the adb hierarchy, measured.** The obvious use for this is to skip
    the ~666ms uiautomator2 reconnect after an offload by re-analyzing from the helper
    instead. Compared element-for-element on two Settings screens, with ``all_windows`` so the
    system bars are included, it does not hold: 54 of 71 elements matched, the helper added
    decorative nodes the XML dump filters (``blue``, ``yellow``), and — the disqualifying one
    — accessibility merges a row into one label, so u2 offers ``"Privacy"`` where the helper
    offers ``"Privacy Permissions, account activity, personal data"``. A selector written
    against one path would silently miss on the other.

    Dropping ``flagIncludeNotImportantViews`` to close the gap makes it worse (28 elements
    against 64: the status bar's internals disappear too).

    So this renders `ui.tree` for inspection and for anything that consumes elements directly.
    Feeding it to `analyze` needs the label-merging difference resolved first, and that is a
    real piece of work, not a flag.
    """

    from xml.sax.saxutils import quoteattr

    def render(node: dict[str, Any], out: list[str]) -> None:
        attrs = []
        for src, dst in _TREE_FIELD_TO_XML.items():
            if src not in node:
                continue
            value = node[src]
            if isinstance(value, bool):
                value = "true" if value else "false"
            attrs.append(f"{dst}={quoteattr(str(value))}")
        children = node.get("children") or []
        if children:
            out.append(f"<node {' '.join(attrs)}>")
            for child in children:
                render(child, out)
            out.append("</node>")
        else:
            out.append(f"<node {' '.join(attrs)}/>")

    parts: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>', '<hierarchy rotation="0">']
    for root in tree.get("roots") or []:
        render(root, parts)
    parts.append("</hierarchy>")
    return "".join(parts)


# --------------------------------------------------------------------------- channel


@dataclass
class HelperChannel:
    """Newline-delimited JSON client for one connected helper.

    Requests are synchronous and correlated by id. Unsolicited events land in a queue the
    caller drains with :meth:`events`, so a slow consumer cannot stall request handling.
    """

    serial: str
    sock: socket.socket
    local_port: int
    _rx: Any = None
    _pending: dict[int, Any] = field(default_factory=dict)
    _events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _wake: threading.Condition = field(init=False)
    _next_id: int = 1
    _closed: bool = False

    def __post_init__(self) -> None:
        self._wake = threading.Condition(self._lock)
        self._rx = threading.Thread(target=self._read_loop, name="aua-helper-rx", daemon=True)
        self._rx.start()

    def _read_loop(self) -> None:
        buf = b""
        try:
            while not self._closed:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    with self._wake:
                        if msg.get("event"):
                            self._events.append(msg)
                        elif msg.get("id") is not None:
                            self._pending[int(msg["id"])] = msg
                        self._wake.notify_all()
        except OSError:
            pass
        finally:
            with self._wake:
                self._closed = True
                self._wake.notify_all()

    def request(self, method: str, params: dict[str, Any] | None = None, *, timeout: float = 5.0):
        with self._wake:
            req_id = self._next_id
            self._next_id += 1
        payload = json.dumps({"id": req_id, "method": method, "params": params or {}}) + "\n"
        self.sock.sendall(payload.encode())

        deadline = time.monotonic() + timeout
        with self._wake:
            while req_id not in self._pending:
                if self._closed:
                    raise HelperUnavailableError(
                        "helper channel closed while waiting for a reply",
                        code="helper_channel_closed",
                    )
                if not self._wake.wait(max(0.0, deadline - time.monotonic())):
                    raise HelperUnavailableError(
                        f"helper did not answer {method} within {timeout:.1f}s",
                        code="helper_timeout",
                    )
            msg = self._pending.pop(req_id)
        if not msg.get("ok"):
            raise HelperUnavailableError(
                f"helper rejected {method}: {msg.get('error')}", code="helper_error"
            )
        return msg.get("result") or {}

    def subscribe(self, types: list[int] | None = None) -> dict[str, Any]:
        return self.request("a11y.subscribe", {"types": types} if types is not None else {})

    def events(self, *, timeout: float) -> Iterator[dict[str, Any]]:
        """Yield queued events, waiting up to *timeout* for the first one."""

        deadline = time.monotonic() + timeout
        with self._wake:
            while not self._events and not self._closed:
                if not self._wake.wait(max(0.0, deadline - time.monotonic())):
                    break
            drained, self._events = self._events, []
        yield from drained

    def wait_for_event(self, *, timeout: float) -> dict[str, Any] | None:
        for event in self.events(timeout=timeout):
            return event
        return None

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        finally:
            subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "forward", "--remove", f"tcp:{self.local_port}"],
                capture_output=True,
                check=False,
                timeout=10,
            )


def _connect_channel(serial: str, *, timeout: float = 5.0) -> HelperChannel:
    """Forward a port and connect, without judging readiness first.

    Split out from :func:`open_channel` so ``status`` can ask the helper whether its service
    is attached; going through the readiness check would be circular, since the channel *is*
    the authority that check wants to consult.
    """

    # tcp:0 lets adb allocate: several agents drive several devices from one host, and a
    # fixed port would make them collide.
    forwarded = _adb(serial, "forward", "tcp:0", f"tcp:{DEVICE_PORT}", timeout=15)
    raw = (forwarded.stdout or "").strip()
    if not raw.isdigit():
        raise HelperUnavailableError(
            f"could not forward a port to the helper on {serial}: {raw or 'no port returned'}",
            code="helper_forward_failed",
        )
    local_port = int(raw)
    try:
        sock = socket.create_connection(("127.0.0.1", local_port), timeout=timeout)
        sock.settimeout(None)
    except OSError as exc:
        _adb(serial, "forward", "--remove", f"tcp:{local_port}", timeout=10)
        raise HelperUnavailableError(
            f"helper channel refused the connection on {serial}: {exc}",
            code="helper_connect_failed",
        ) from exc

    return HelperChannel(serial=serial, sock=sock, local_port=local_port)


def open_channel(serial: str, *, timeout: float = 5.0) -> HelperChannel:
    """Connect to a ready helper and verify the wire contract before handing it over."""

    channel = _connect_channel(serial, timeout=timeout)
    try:
        info = channel.request("helper.info", timeout=timeout)
    except HelperUnavailableError:
        channel.close()
        raise
    if int(info.get("protocol", 0)) != PROTOCOL:
        channel.close()
        raise HelperUnavailableError(
            f"helper on {serial} speaks protocol {info.get('protocol')}, this AUA needs {PROTOCOL}",
            code="helper_protocol_mismatch",
            hint="Run `aua helper install --reinstall` to refresh the on-device helper.",
        )
    if not info.get("service_bound", True):
        channel.close()
        raise HelperUnavailableError(
            f"the helper on {serial} is installed but its accessibility service is not attached",
            code="helper_not_running",
            hint="Run `aua helper enable` (needs a rootable target).",
        )
    if not _can_see_the_screen(channel, timeout=timeout):
        channel.close()
        raise HelperUnavailableError(
            f"the helper on {serial} is attached but cannot read the screen yet",
            code="helper_not_ready",
            hint="Usually transient: the helper process was frozen and is still thawing.",
        )
    return channel


def _can_see_the_screen(channel: HelperChannel, *, timeout: float) -> bool:
    """Ask the helper to prove it can read the screen, rather than trusting a status flag.

    Neither ``dumpsys accessibility`` nor the service's own ``service_bound`` is a readiness
    signal, and believing them is what made the offload unreliable. Android freezes the helper
    process whenever the service is torn down — which is *most* of AUA's life, because
    uiautomator2 holding UiAutomation is exactly what tears it down. When the slot is handed
    back, the framework lists the service as bound before the process has finished thawing, so
    both flags say yes while ``getRootInActiveWindow`` still returns null. Steps then fail one
    after another with "no node matched", on a screen that plainly has the node.

    Asking for the tree is the only question whose answer cannot be stale, because it is the
    same call every step depends on. A no here costs one slower run and never a wrong one.
    """

    try:
        tree = channel.request("ui.tree", None, timeout=timeout)
    except HelperUnavailableError:
        return False
    roots = tree.get("roots") if isinstance(tree, dict) else None
    return bool(roots)


# -- raw touch capture ------------------------------------------------------
#
# Accessibility only reports a tap when the view bothers to announce one, and plenty do not.
# On one real Compose app, NONE did: seventeen of nineteen recorded steps came from here. The
# kernel input stream has no such opinion — every finger that touches the glass appears in it.
# That is what lets a recording name the button behind a tap Android stayed silent about.
#
# Note for anyone testing this: `adb shell input tap` is injected above the kernel and does
# NOT appear here. Only a real finger or a synthetic `sendevent` does.

TOUCH_LOG = "/data/local/tmp/aua-touches.log"

# getevent -lt line, e.g.
#   [   48361.896424] /dev/input/event3: EV_ABS  ABS_MT_POSITION_X  00003fff
_EVENT_LINE = re.compile(
    r"^\[\s*(?P<ts>\d+\.\d+)\]\s+(?P<dev>/dev/input/event\d+):\s+"
    r"(?P<type>\w+)\s+(?P<code>\w+)\s+(?P<value>[0-9a-fA-F]+)"
)
# The clock line this module writes into the log before anything else, so the monotonic event
# stamps can be turned back into wall time by whoever reads the log later.
_CLOCK_LINE = re.compile(r"^#clock\s+(?P<uptime>[\d.]+)\s+(?P<wall_ms>\d+)")

# Separates the device listing from the event stream in the capture.
_EVENTS_MARKER = "#events"

# A press that travels further than this is a drag, not a tap. Generous: a finger on glass
# always moves a little, and treating that wobble as a swipe would discard the very taps this
# exists to recover. Checked against a real 19-step journey — nothing was misread.
_TAP_SLOP_PX = 40


@dataclass(frozen=True)
class Touch:
    """One finger-down/up gesture, in screen pixels and wall-clock milliseconds."""

    x: int
    y: int
    down_ms: int
    up_ms: int
    travel_px: int

    @property
    def is_tap(self) -> bool:
        return self.travel_px <= _TAP_SLOP_PX


def parse_device_axes(listing: str) -> dict[str, tuple[int, int]]:
    """Per-device (max_x, max_y) for the multi-touch axes, from a ``getevent -pl`` listing.

    Read out of the capture rather than by asking the device again, so it describes the device
    as it was when recording started.

    The listing also carries a ``value`` per axis, which would be the obvious way to know the
    state a first press inherits. It cannot be trusted — emulators report 0 there regardless of
    where the last touch actually was — so a press whose position was never established is
    dropped instead. See :func:`parse_touch_log`.
    """

    out: dict[str, tuple[int, int]] = {}
    device: str | None = None
    x_max = y_max = 0

    def flush() -> None:
        if device and x_max and y_max:
            out[device] = (x_max, y_max)

    for line in listing.splitlines():
        stripped = line.strip()
        if stripped.startswith("add device"):
            flush()
            x_max = y_max = 0
            _, _, path = stripped.partition(":")
            device = path.strip() or None
        elif "ABS_MT_POSITION_X" in stripped:
            bounds = re.search(r"max (\d+)", stripped)
            if bounds:
                x_max = int(bounds.group(1))
        elif "ABS_MT_POSITION_Y" in stripped:
            bounds = re.search(r"max (\d+)", stripped)
            if bounds:
                y_max = int(bounds.group(1))
    flush()
    return out


def parse_touch_log(
    text: str, *, axis_maxima: dict[str, tuple[int, int]], screen: tuple[int, int]
) -> list[Touch]:
    """Turn a getevent capture into finger gestures in screen pixels.

    Pure, so it can be tested against a recorded capture without a device attached — which
    matters more than usual here, because the one thing that cannot be simulated on this path
    is a touch (``input tap`` bypasses the kernel entirely).

    Two rules of the input protocol drive the whole shape of this:

    *SYN_REPORT is where a state becomes real.* X and Y arrive as separate events, so neither
    alone is a position; the kernel commits a complete state at SYN. Reading the position when
    the tracking id opens the gesture instead gives whatever the *previous* finger left behind.

    *An axis that has not changed is not re-sent.* So a second press in the same place emits no
    position at all, and its coordinates are simply the ones already standing. That is why a
    position carries forward between gestures rather than resetting — but also why a press that
    arrives before either axis has ever reported is DROPPED rather than reported at the origin.
    A dropped press leaves an honest hole in the recording; a press invented at (0, 0) resolves
    to whatever sits at the top of the screen and names the wrong button with total confidence.
    """

    width, height = screen
    offset_ms: float | None = None
    # Per input device: the standing position, which axes have ever reported, and the gesture
    # in progress.
    last: dict[str, list[int]] = {}
    established: dict[str, set[str]] = {}
    live: dict[str, dict[str, Any]] = {}
    touches: list[Touch] = []

    for raw in text.splitlines():
        clock = _CLOCK_LINE.match(raw.strip())
        if clock:
            # wall_ms at the instant uptime was read: everything else is uptime + this.
            offset_ms = int(clock.group("wall_ms")) - float(clock.group("uptime")) * 1000.0
            continue
        match = _EVENT_LINE.match(raw)
        if not match or offset_ms is None:
            continue
        device = match.group("dev")
        code = match.group("code")
        value = int(match.group("value"), 16)
        when_ms = int(float(match.group("ts")) * 1000.0 + offset_ms)
        maxima = axis_maxima.get(device)
        if maxima is None:
            continue
        max_x, max_y = maxima
        point = last.setdefault(device, [0, 0])
        seen = established.setdefault(device, set())

        if code == "ABS_MT_POSITION_X":
            point[0] = round(value / max_x * width) if max_x else value
            seen.add("x")
        elif code == "ABS_MT_POSITION_Y":
            point[1] = round(value / max_y * height) if max_y else value
            seen.add("y")
        elif code == "SYN_REPORT":
            gesture = live.get(device)
            if gesture is not None and gesture["start"] is None and {"x", "y"} <= seen:
                gesture["start"] = list(point)
        elif code == "ABS_MT_TRACKING_ID":
            if value != 0xFFFFFFFF:
                live[device] = {"down_ms": when_ms, "start": None}
            else:
                started = live.pop(device, None)
                if started is None or started["start"] is None:
                    continue   # a press whose position was never established
                start_point = started["start"]
                travel = max(
                    abs(point[0] - start_point[0]), abs(point[1] - start_point[1])
                )
                touches.append(
                    Touch(
                        x=int(start_point[0]),
                        y=int(start_point[1]),
                        down_ms=int(started["down_ms"]),
                        up_ms=when_ms,
                        travel_px=int(travel),
                    )
                )
    return touches


def start_touch_capture(serial: str) -> dict[str, Any]:
    """Begin recording raw touches on the device, surviving this host process exiting.

    Runs on the device rather than as a host subprocess precisely because ``aua demo start``
    and ``aua demo stop`` are two separate short-lived commands with a person's whole journey
    in between; there is no host process alive to own a pipe across that.

    The clock line and the device listing are written into the log ahead of the stream, so
    everything needed to interpret it travels with it, in whatever process reads it later.
    """

    _shell(serial, f"rm -f {TOUCH_LOG}", timeout=15.0)
    # `read` rather than `cut`, because this string is nested three shells deep (host -> adb ->
    # sh -c) and every extra quote is somewhere for it to come apart. It did: a quoted
    # `cut -d' '` lost its delimiter and wrote both uptime columns into the clock line.
    script = (
        "read up rest < /proc/uptime; "
        f'{{ echo "#clock $up $(date +%s%3N)"; getevent -pl; echo "{_EVENTS_MARKER}"; '
        f"getevent -lt; }} > {TOUCH_LOG} 2>&1"
    )
    # Detached deliberately: it must outlive this adb shell and every command after it, because
    # the journey being recorded happens between two separate short-lived CLI invocations.
    _shell(serial, f"nohup sh -c '{script}' >/dev/null 2>&1 &", timeout=15.0)
    return {"capturing": True, "log": TOUCH_LOG}


def _screen_size(serial: str) -> tuple[int, int] | None:
    """Display size in pixels, read without connecting uiautomator2.

    Connecting would take the UiAutomation slot and tear down the accessibility service, which
    is the one thing every command on this path is careful not to do.
    """

    try:
        out = _shell(serial, "wm size", timeout=15.0)
    except Exception:  # noqa: BLE001 - no size means no unnormalizing; caller copes
        return None
    match = re.search(r"Override size:\s*(\d+)x(\d+)", out) or re.search(
        r"Physical size:\s*(\d+)x(\d+)", out
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def stop_touch_capture(serial: str) -> list[Touch]:
    """Stop recording and return the gestures. Never raises: no touches is a valid answer.

    Everything here is best effort by design. The touch stream is an *extra* source used to
    explain taps accessibility missed, so a device that will not give it up leaves the
    recording exactly as good as it was before — holes reported honestly — rather than failing
    a journey somebody already walked.
    """

    with contextlib.suppress(Exception):
        _shell(serial, "pkill -f 'getevent -lt'", timeout=15.0)
    try:
        text = _shell(serial, f"cat {TOUCH_LOG} 2>/dev/null", timeout=30.0)
    except Exception:  # noqa: BLE001 - see docstring
        return []
    with contextlib.suppress(Exception):
        _shell(serial, f"rm -f {TOUCH_LOG}", timeout=15.0)
    screen = _screen_size(serial)
    if screen is None:
        return []
    listing, _, _stream = text.partition(_EVENTS_MARKER)
    try:
        # The whole text is parsed for events, not just the half after the marker: the clock
        # line is written before the listing, and without it every stamp is uninterpretable.
        # Listing lines are not event-shaped, so they are ignored rather than misread.
        return parse_touch_log(text, axis_maxima=parse_device_axes(listing), screen=screen)
    except Exception:  # noqa: BLE001 - see docstring
        return []
