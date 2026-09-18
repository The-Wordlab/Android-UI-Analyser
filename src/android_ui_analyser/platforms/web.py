"""Built-in web platform: Playwright browser pages through AUA's semantic runtime."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from ..config import Config
from ..errors import ConfigError, DeviceError, UsageError
from ..providers.base import ScreenImage
from ..schema import TargetInfo, TargetStatus
from . import web_tree
from .base import DiscoveredTarget, NormalizedTree, PlatformAdapter
from .geometry import DisplayGeometry
from .registry import register_platform
from .runtime import TargetRuntime
from .web_runtime import KEY_NAMES, WebRuntime
from .web_tools import PlaywrightLauncher, WebLauncher, WebLaunchOptions

_BROWSERS = frozenset({"chromium", "firefox", "webkit"})


def _url(value: Any, *, field: str = "url") -> str:
    candidate = str(value or "").strip()
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(
            f"web {field} must be an absolute http(s) URL",
            hint="Example: https://example.test/app",
        )
    if parsed.username or parsed.password:
        raise ConfigError(
            f"web {field} must not contain credentials",
            hint="Seed authenticated state with platforms.web.storage_state instead.",
        )
    return candidate


def _optional_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        raise ConfigError(f"web option {field} must be a non-empty string")
    return candidate


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"web option {field} must be a positive integer")
    return value


@register_platform("web")
class WebPlatform(PlatformAdapter):
    """One isolated browser page, with DOM semantics normalized to AUA elements."""

    capabilities = frozenset({"app.links", "ui.input", "ui.screenshot", "ui.tree"})

    def __init__(self, config: Config, launcher: WebLauncher | None = None) -> None:
        super().__init__(config)
        self._launcher = launcher or PlaywrightLauncher()
        self._uses_default_launcher = launcher is None

    def validate_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        known = {
            "url",
            "browser",
            "headless",
            "channel",
            "executable_path",
            "viewport_width",
            "viewport_height",
            "navigation_timeout_ms",
            "action_timeout_ms",
            "storage_state",
            "ignore_https_errors",
        }
        unknown = sorted(str(key) for key in options if key not in known)
        if unknown:
            raise ConfigError(
                f"platform 'web' does not accept options: {', '.join(unknown)}",
                hint="See docs/web.md for the supported browser options.",
            )
        normalized: dict[str, Any] = {}
        if "url" in options:
            normalized["url"] = _url(options["url"])
        browser = str(options.get("browser", "chromium")).strip().casefold()
        if browser not in _BROWSERS:
            raise ConfigError(
                f"unknown web browser {browser!r}",
                hint="Choose chromium, firefox, or webkit.",
            )
        normalized["browser"] = browser
        for field in ("headless", "ignore_https_errors"):
            value = options.get(field, field == "headless")
            if not isinstance(value, bool):
                raise ConfigError(f"web option {field} must be true or false")
            normalized[field] = value
        for field, default in (
            ("viewport_width", 1280),
            ("viewport_height", 800),
            ("navigation_timeout_ms", 30_000),
            ("action_timeout_ms", 5_000),
        ):
            normalized[field] = _positive_int(options.get(field, default), field=field)
        for field in ("channel", "executable_path", "storage_state"):
            value = _optional_text(options.get(field), field=field)
            if value is not None:
                normalized[field] = value
        if browser != "chromium" and "channel" in normalized:
            raise ConfigError("web option channel is supported only by chromium")
        return normalized

    def _launch_options(self) -> WebLaunchOptions:
        values = dict(self.options)
        values.pop("url", None)
        return WebLaunchOptions(**values)

    def prepare_host(self) -> None:
        if self._uses_default_launcher and importlib.util.find_spec("playwright") is None:
            raise DeviceError(
                "web support needs the optional Playwright dependency",
                code="web_driver_missing",
                hint="Install `android-ui-analyser[web]`, then run `playwright install chromium`.",
            )

    def list_targets(self) -> list[DiscoveredTarget]:
        candidate = self.options.get("url") or self.config.device.serial
        if not candidate:
            return []
        target = _url(candidate, field="target")
        return [
            TargetInfo(
                target_id=target,
                platform=self.name,
                status=TargetStatus.online,
                model=str(self.options.get("browser", "chromium")),
                os_name="web",
            )
        ]

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        target = target_id or self.options.get("url")
        if not target:
            raise DeviceError(
                "no web URL configured",
                code="no_target",
                hint="Pass `--serial https://…` or set platforms.web.url in AUA config.",
            )
        url = _url(target, field="target")
        self.prepare_host()
        return WebRuntime(self._launcher.launch(url, self._launch_options()), url)

    def normalize_key(self, name: str) -> str:
        candidate = super().normalize_key(name).casefold()
        if candidate not in KEY_NAMES:
            raise UsageError(
                f"unknown web key {name!r}",
                hint="Valid: " + ", ".join(sorted(KEY_NAMES)) + ".",
            )
        return candidate

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        del geometry
        return web_tree.normalize(raw_tree, screen_size, ignored_app_ids=ignored_app_ids)

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        return runtime.screenshot()

    def doctor_checks(self) -> dict[str, Any]:
        installed = importlib.util.find_spec("playwright") is not None
        configured = self.options.get("url") or self.config.device.serial
        return {
            "platform": {
                "ok": True,
                "detail": self.name,
                "capabilities": sorted(self.capabilities),
            },
            "playwright": {
                "ok": installed,
                "detail": "installed" if installed else "not installed",
                **(
                    {}
                    if installed
                    else {
                        "hint": "Install `android-ui-analyser[web]`, then run `playwright install chromium`."
                    }
                ),
            },
            "target": {
                "ok": bool(configured),
                "detail": str(configured) if configured else "no URL configured",
                **(
                    {}
                    if configured
                    else {"hint": "Pass --serial https://… or set platforms.web.url."}
                ),
            },
        }


__all__ = ["WebPlatform"]
