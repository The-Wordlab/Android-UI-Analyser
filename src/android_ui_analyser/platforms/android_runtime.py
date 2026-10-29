"""Android UI runtime recovery owned by :class:`AndroidPlatform`.

The shared engine sees only the platform-neutral ``Device`` contract. Android's one-process
UiAutomation registration and its teardown commands stay here, behind the selected adapter.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator
from typing import Any

from ..errors import ConfigError
from .android_device import Uiautomator2Device

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

    _development_tools: Any = None

    def __init__(
        self, serial: str, settle_wait: float = 0.0, *, development_tools: Any = None
    ) -> None:
        self._development_tools = development_tools
        super().__init__(serial, settle_wait)

    def _development_service(self) -> Any:
        if self._development_tools is None:
            from .android_cli import AndroidCliTools

            self._development_tools = AndroidCliTools()
        return self._development_tools

    def execute_development_tool(
        self,
        operation: str,
        arguments: list[str],
        options: dict[str, Any],
        project_dir: str | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        tools = self._development_service()
        tools.require_available()
        # Engine has durably recorded the helper undo and stood down UiAutomation
        # before calling this runtime. It restores the helper before reconnecting.
        if operation == "profiler.session.terminate":
            return tools.terminate_profile(options["session"], self.serial)
        return tools.execute(
            operation,
            arguments,
            options,
            project_dir,
            timeout_s,
            target_id=self.serial,
        )

    def restore_development_tool(self, session_id: str) -> dict[str, Any]:
        return self._development_service().terminate_profile(session_id, self.serial)

    def prepare_development_ui(self) -> dict[str, Any]:
        return self._development_service().prepare_ui_helper(self.serial)

    def restore_development_ui(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._development_service().release_ui_helper(self.serial, state)

    def _connect(self) -> None:
        from ..device_ledger import ledger_dir, read_ledger
        from .options_transport import platform_options_fingerprint

        # A dead launcher can leave the helper holding Android's only UiAutomation
        # slot. Replay only this exact AUA-recorded undo before uiautomator2 connects;
        # generic connection recovery must never stop an unrelated tool's helper.
        for entry in read_ledger(self.serial):
            if entry.kind == "development_tool_ui" and entry.op == "restore_development_ui":
                fingerprint = platform_options_fingerprint(
                    self._development_service().options, key_dir=ledger_dir()
                )
                if entry.platform_options_fingerprint != fingerprint:
                    raise ConfigError(
                        "Pending developer UI cleanup belongs to different platform options",
                        code="device_ledger_options_mismatch",
                    )
                state = entry.args.get("state")
                if not isinstance(state, dict):
                    raise ConfigError("Pending developer UI cleanup has invalid restoration state")
                if self.restore_development_ui(state).get("ok") is not True:
                    raise ConfigError("Pending developer UI helper restoration was unsuccessful")
        super()._connect()

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
