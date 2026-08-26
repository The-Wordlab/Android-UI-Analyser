"""Android UI runtime recovery owned by :class:`AndroidPlatform`.

The shared engine sees only the platform-neutral ``Device`` contract. Android's one-process
UiAutomation registration and its teardown commands stay here, behind the selected adapter.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator

from ..device import Uiautomator2Device

_STALE_UIAUTOMATION_MARKERS = (
    "not connected",
    "already connected",
    "already registered",
)


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def stale_uiautomation_error(error: BaseException) -> bool:
    """Recognize Android's stale single-slot UiAutomation failures conservatively."""

    detail = " | ".join(str(item) for item in _exception_chain(error)).casefold()
    return "uiautomation" in detail and any(
        marker in detail for marker in _STALE_UIAUTOMATION_MARKERS
    )


class AndroidDeviceRuntime(Uiautomator2Device):
    """uiautomator2 runtime with one serial-scoped stale-registration recovery."""

    def _recover_connection(self, name: str, error: Exception) -> None:
        if stale_uiautomation_error(error):
            client = self._d
            self._d = None
            # A client that attached to a server created by an earlier process often has no
            # subprocess handle to stop. Kill the Android-side server by name, scoped to this
            # leased serial, before reconnecting. Another emulator is never addressed.
            with contextlib.suppress(Exception):
                subprocess.run(  # noqa: S603
                    [
                        "adb",
                        "-s",
                        self.serial,
                        "shell",
                        "pkill",
                        "-f",
                        "com.wetest.uia2.Main",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=15,
                )
            if client is not None:
                with contextlib.suppress(Exception):
                    client.stop_uiautomator()
        self._connect()

