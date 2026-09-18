"""Access confined to an installed simulator app's data container."""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import DeviceError, UsageError
from .ios_tools import IOSTools


class IOSAppFiles:
    def __init__(self, tools: IOSTools, target_id: str) -> None:
        self.tools = tools
        self.target_id = target_id

    def root(self, app_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", app_id):
            raise UsageError("invalid app bundle identifier")
        result = self.tools.simctl("get_app_container", self.target_id, app_id, "data", check=False)
        path = Path(result.text.strip()).resolve()
        if (
            not result.ok
            or not path.is_dir()
            or "/Containers/Data/Application/" not in path.as_posix()
        ):
            raise DeviceError("installed app data container unavailable", code="app_not_installed")
        return path

    def path(self, app_id: str, relative: str) -> Path:
        root = self.root(app_id)
        requested = Path(relative)
        if requested.is_absolute() or ".." in requested.parts or not requested.parts:
            raise UsageError("app file must be a relative path inside its data container")
        path = (root / requested).resolve()
        if not path.is_relative_to(root) or path == root:
            raise UsageError("app file escapes its data container")
        return path

    def remove(self, app_id: str, paths: list[str]) -> None:
        resolved = [self.path(app_id, relative) for relative in paths]
        if any(path.is_dir() for path in resolved):
            raise UsageError("app file removal does not accept directories")
        for path in resolved:
            path.unlink(missing_ok=True)
