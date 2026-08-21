"""Publish the dashboard under a typeable ``<name>.local`` mDNS hostname.

The dashboard's honest URL is ``http://192.168.8.240:48765/?token=…``. Nobody types
that. This module registers a multicast-DNS host record so the same page answers to
``http://aua.local/`` — a name a person can type into a browser from the host machine
or from any other device on the same private network.

Two properties make this safe to switch on by default for a LAN dashboard:

* **No privilege.** ``dns-sd`` (macOS) and ``avahi-publish`` (Linux) both register a
  host record as an ordinary user. Nothing is written to ``/etc/hosts``, no port below
  1024 is claimed on the loopback interface, and no password is ever requested.
* **No residue.** The record lives inside the publisher process. When the dashboard
  exits the child is terminated and the name evaporates from the network, so a crashed
  or killed dashboard cannot leave a stale name pointing at nothing.

Resolution support is a property of the *client*, not of this record: Apple platforms
and Windows 10+ resolve ``.local`` natively, while Android's browsers do not. The QR
code carrying a raw IP therefore remains the phone path and is not replaced by this.
"""

from __future__ import annotations

import contextlib
import logging
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import UsageError

logger = logging.getLogger(__name__)

DEFAULT_HOSTNAME = "aua"
MDNS_SUFFIX = ".local"

# One DNS label: letters, digits and inner hyphens, at most 63 characters. Deliberately
# stricter than RFC 1123 (no leading digit-only edge cases to reason about) because the
# whole point of the name is that a human types it from memory.
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

# How long to wait for the freshly published record to answer a lookup. Registration is
# a multicast round trip plus probing for conflicts; on a quiet network it lands well
# inside a second, and a slow one is reported as unresolved rather than silently assumed.
_RESOLVE_TIMEOUT_S = 3.0
_RESOLVE_INTERVAL_S = 0.15


def normalise_hostname(name: str) -> str:
    """Return the bare label for ``name``, rejecting anything unusable as a hostname.

    Accepts ``aua``, ``AUA`` and ``aua.local`` alike so a user who copies the URL back
    out of the browser gets the same answer as one who types the short form.
    """
    label = str(name or "").strip().lower().rstrip(".")
    if label.endswith(MDNS_SUFFIX):
        label = label[: -len(MDNS_SUFFIX)]
    if not label:
        raise UsageError(
            "dashboard hostname is empty",
            hint=f"Pass a single name such as `--name {DEFAULT_HOSTNAME}`.",
        )
    if not _LABEL.match(label):
        raise UsageError(
            f"{name!r} is not usable as an mDNS hostname",
            hint="Use letters, digits and inner hyphens only — one label, no dots.",
        )
    return label


def hostname_for(name: str) -> str:
    """Return the fully qualified mDNS name, e.g. ``aua`` -> ``aua.local``."""
    return normalise_hostname(name) + MDNS_SUFFIX


def hostname_url(name: str, port: int) -> str:
    """Return the browser URL for ``name``, omitting the default HTTP port."""
    host = hostname_for(name)
    return f"http://{host}/" if int(port) == 80 else f"http://{host}:{int(port)}/"


def publisher_command(
    *,
    hostname: str,
    port: int,
    address: str,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str] | None:
    """Build the argv that publishes ``hostname`` -> ``address``, or ``None`` if unsupported.

    ``platform`` and ``which`` are injected so the mapping can be asserted on any host
    without a publisher binary installed.
    """
    fqdn = hostname_for(hostname)
    system = platform if platform is not None else sys.platform
    if system == "darwin":
        tool = which("dns-sd")
        if not tool:
            return None
        # ``-P`` registers a *proxy* record: a service advertisement plus the A record that
        # makes ``fqdn`` resolve. Plain ``-R`` advertises only the service, which Bonjour
        # browsers see but a browser address bar cannot resolve.
        return [tool, "-P", normalise_hostname(hostname), "_http._tcp", "local",
                str(int(port)), fqdn, address]
    if system.startswith("linux"):
        tool = which("avahi-publish")
        if not tool:
            return None
        # ``-a`` publishes an address record; ``-R`` allows it to coexist with the records
        # avahi already owns for this machine's own hostname.
        return [tool, "-a", "-R", fqdn, address]
    return None


def resolves_to(hostname: str, address: str | None = None) -> bool:
    """True when ``hostname`` resolves, optionally to ``address`` specifically."""
    fqdn = hostname_for(hostname)
    try:
        infos = socket.getaddrinfo(fqdn, None, socket.AF_INET)
    except OSError:
        return False
    found = {str(info[4][0]) for info in infos}
    return bool(found) and (address is None or address in found)


@dataclass
class Advertisement:
    """A live mDNS host record, owned by a child process that dies with the dashboard."""

    hostname: str
    address: str
    port: int
    url: str
    resolved: bool
    _process: subprocess.Popen[bytes] | None = None

    def stop(self) -> None:
        """Terminate the publisher so the name leaves the network immediately."""
        proc = self._process
        self._process = None
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=1.0)

    def info(self) -> dict[str, Any]:
        return {
            "hostname": hostname_for(self.hostname),
            "address": self.address,
            "port": int(self.port),
            "url": self.url,
            "resolved": bool(self.resolved),
        }


def advertise(
    *,
    hostname: str,
    port: int,
    address: str,
    spawn: Callable[[list[str]], subprocess.Popen[bytes]] | None = None,
    command: list[str] | None = None,
    wait: bool = True,
) -> Advertisement | None:
    """Publish ``hostname.local`` -> ``address`` for the lifetime of the returned handle.

    Returns ``None`` when the host has no usable publisher, which is a soft failure: the
    dashboard is perfectly usable at its IP and must not refuse to start over a
    convenience name.
    """
    argv = command if command is not None else publisher_command(
        hostname=hostname, port=port, address=address
    )
    if not argv:
        logger.info("no mDNS publisher on this host; dashboard keeps its IP URL only")
        return None

    launch = spawn or (
        lambda cmd: subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    )
    try:
        proc = launch(argv)
    except OSError as exc:
        logger.info("mDNS publisher failed to start (%s); keeping the IP URL", exc)
        return None

    resolved = False
    if wait:
        deadline = time.monotonic() + _RESOLVE_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                logger.info("mDNS publisher exited early; keeping the IP URL")
                break
            if resolves_to(hostname, address):
                resolved = True
                break
            time.sleep(_RESOLVE_INTERVAL_S)

    return Advertisement(
        hostname=normalise_hostname(hostname),
        address=address,
        port=int(port),
        url=hostname_url(hostname, port),
        resolved=resolved,
        _process=proc,
    )
