"""Built-in web platform: Playwright browser pages through AUA's semantic runtime."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from ..config import Config
from ..errors import ConfigError, DeviceError, UsageError
from ..providers.base import ScreenImage
from ..schema import TargetInfo, TargetStatus
from . import web_tree
from .base import DiscoveredTarget, NormalizedTree, PlatformAdapter
from .diagnostics import (
    DiagnosticEvent,
    DiagnosticLevel,
    DiagnosticWindow,
    UnknownDiagnosticMark,
)
from .geometry import DisplayGeometry
from .identity import TargetRef
from .registry import register_platform
from .runtime import TargetRuntime
from .web_runtime import KEY_NAMES, WebRuntime
from .web_tools import PlaywrightLauncher, WebLauncher, WebLaunchOptions

_BROWSERS = frozenset({"chromium", "firefox", "webkit"})
_SERVICE_WORKERS = frozenset({"allow", "block"})


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

    capabilities = frozenset(
        {
            "app.links",
            "browser.diagnostics",
            "browser.network",
            "browser.pages",
            "browser.storage",
            "browser.trace",
            "device.logs",
            "session.state",
            "ui.input",
            "ui.screenshot",
            "ui.tree",
        }
    )

    def __init__(self, config: Config, launcher: WebLauncher | None = None) -> None:
        super().__init__(config)
        self._launcher = launcher or PlaywrightLauncher()
        self._uses_default_launcher = launcher is None
        self._runtimes: dict[str, WebRuntime] = {}
        self._diagnostic_marks: dict[tuple[str, str], int] = {}

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
            "bypass_csp",
            "service_workers",
            "proxy_server",
            "proxy_bypass",
            "proxy_username",
            "proxy_password_env",
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
        for field in ("headless", "ignore_https_errors", "bypass_csp"):
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
        for field in (
            "channel",
            "executable_path",
            "storage_state",
            "proxy_server",
            "proxy_bypass",
            "proxy_username",
            "proxy_password_env",
        ):
            value = _optional_text(options.get(field), field=field)
            if value is not None:
                normalized[field] = value
        service_workers = str(options.get("service_workers", "allow")).strip().casefold()
        if service_workers not in _SERVICE_WORKERS:
            raise ConfigError("web option service_workers must be 'allow' or 'block'")
        normalized["service_workers"] = service_workers
        if "proxy_server" in normalized:
            parsed_proxy = urlsplit(normalized["proxy_server"])
            if (
                parsed_proxy.scheme not in {"http", "https", "socks4", "socks5"}
                or not parsed_proxy.hostname
            ):
                raise ConfigError(
                    "web option proxy_server must be an absolute http(s), socks4, or socks5 URL"
                )
            if parsed_proxy.username or parsed_proxy.password:
                raise ConfigError(
                    "web proxy credentials must use proxy_username and proxy_password_env"
                )
        if browser != "chromium" and "channel" in normalized:
            raise ConfigError("web option channel is supported only by chromium")
        return normalized

    def _launch_options(self) -> WebLaunchOptions:
        values = dict(self.options)
        values.pop("url", None)
        password_env = values.pop("proxy_password_env", None)
        if password_env is not None:
            password = os.environ.get(str(password_env))
            if password is None:
                raise ConfigError(
                    f"web proxy password environment variable {password_env!r} is not set"
                )
            values["proxy_password"] = password
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
        runtime = WebRuntime(self._launcher.launch(url, self._launch_options()), url)
        self._runtimes[url] = runtime
        return runtime

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

    @staticmethod
    def _diagnostic_level(value: object) -> DiagnosticLevel | None:
        return {
            "verbose": DiagnosticLevel.VERBOSE,
            "debug": DiagnosticLevel.DEBUG,
            "log": DiagnosticLevel.INFO,
            "info": DiagnosticLevel.INFO,
            "warning": DiagnosticLevel.WARNING,
            "warn": DiagnosticLevel.WARNING,
            "error": DiagnosticLevel.ERROR,
            "assert": DiagnosticLevel.ERROR,
            "fatal": DiagnosticLevel.FATAL,
        }.get(str(value).casefold())

    def diagnostic_window(
        self,
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since: str | int | None = None,
        app_id: str | None = None,
    ) -> DiagnosticWindow:
        browser = self.runtime_capability("browser.diagnostics", runtime)
        since_ms: int | None
        since_label: str | None
        if isinstance(since, str):
            key = (runtime.target_id, since)
            if key not in self._diagnostic_marks:
                known = [
                    name
                    for (target_id, name), _value in self._diagnostic_marks.items()
                    if target_id == runtime.target_id
                ]
                raise UnknownDiagnosticMark(since, known)
            since_ms = self._diagnostic_marks[key]
            since_label = since
        else:
            since_ms = int(since) if since is not None else None
            since_label = str(since) if since is not None else None
        payload = browser.browser_diagnostics(
            limit=max(1, int(lines)), kinds=(), since_ms=since_ms
        )
        events: list[DiagnosticEvent] = []
        for row in payload.get("events") or []:
            if not isinstance(row, Mapping):
                continue
            url = str(row.get("url") or "")
            host = urlsplit(url).hostname if url else None
            if app_id is not None and host != app_id:
                continue
            kind = str(row.get("kind") or "browser")
            message = str(row.get("message") or "")
            events.append(
                DiagnosticEvent(
                    message=message,
                    level=self._diagnostic_level(row.get("level")),
                    source=kind,
                    timestamp_ms=(
                        int(row["timestamp_ms"])
                        if row.get("timestamp_ms") is not None
                        else None
                    ),
                    app_id=host,
                    display_text=f"{kind} | {message}",
                )
            )
        return DiagnosticWindow(
            events=tuple(events[-max(1, int(lines)) :]),
            target=TargetRef(self.name, runtime.target_id),
            since=since_label,
            since_unix_ms=since_ms,
            clock="host",
        )

    def diagnostic_logs(
        self,
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        return self.diagnostic_window(
            runtime, lines=lines, since=since_ms, app_id=app_id
        ).text

    def mark_diagnostics(
        self,
        runtime: TargetRuntime,
        name: str = "default",
        *,
        clear: bool = False,
        refresh_clock: bool = False,
    ) -> dict[str, object]:
        del refresh_clock
        browser = self.runtime_capability("browser.diagnostics", runtime)
        payload = browser.browser_diagnostics_mark(name or "default", clear=clear)
        timestamp_ms = int(payload["timestamp_ms"])
        self._diagnostic_marks[(runtime.target_id, name or "default")] = timestamp_ms
        return {
            "name": name or "default",
            "unix_ms": timestamp_ms,
            "clock": "host",
        }

    def clear_diagnostics(self, runtime: TargetRuntime) -> None:
        browser = self.runtime_capability("browser.diagnostics", runtime)
        browser.browser_diagnostics_clear()
        self._diagnostic_marks = {
            key: value
            for key, value in self._diagnostic_marks.items()
            if key[0] != runtime.target_id
        }

    def recent_logs(
        self, target_id: str, *, limit: int = 80, app_id: str | None = None
    ) -> list[str]:
        runtime = self._runtimes.get(target_id)
        if runtime is None:
            return []
        return self.diagnostic_window(runtime, lines=limit, app_id=app_id).lines

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
