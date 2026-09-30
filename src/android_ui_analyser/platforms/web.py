"""Built-in web platform: Playwright browser pages through AUA's semantic runtime."""

from __future__ import annotations

import hashlib
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
from .chrome_extension import (
    ATTACHED_TARGET_ID,
    ChromeAttachOptions,
    ChromeExtensionLauncher,
)
from .contracts import normalize_capability
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
from .web_cdp import CdpAttachOptions, CdpLauncher, cdp_endpoint, cdp_page_url, cdp_target_id
from .web_runtime import KEY_NAMES, WebRuntime
from .web_tools import PlaywrightLauncher, WebLauncher, WebLaunchOptions

_BROWSERS = frozenset({"chromium", "firefox", "webkit"})
_SERVICE_WORKERS = frozenset({"allow", "block"})
_CONNECTIONS = frozenset({"isolated", "existing-chrome", "existing-cdp"})
_ATTACHED_UNSAFE_CAPABILITIES = frozenset(
    {"browser.network", "browser.storage", "browser.trace"}
)


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
    """One isolated or explicitly approved browser page normalized to AUA elements."""

    capabilities = frozenset(
        {
            "app.links",
            "browser.diagnostics",
            "browser.network",
            "browser.pages",
            "browser.storage",
            "browser.trace",
            "device.drag",
            "device.logs",
            "session.state",
            "ui.input",
            "ui.read_deadline",
            "ui.screenshot",
            "ui.tree",
        }
    )

    def __init__(
        self,
        config: Config,
        launcher: WebLauncher | None = None,
        extension_launcher: Any | None = None,
        cdp_launcher: CdpLauncher | None = None,
    ) -> None:
        super().__init__(config)
        self._launcher = launcher or PlaywrightLauncher()
        self._extension_launcher = extension_launcher or ChromeExtensionLauncher()
        self._cdp_launcher = cdp_launcher or CdpLauncher()
        self._uses_default_cdp_launcher = cdp_launcher is None
        self._uses_default_launcher = launcher is None
        self._runtimes: dict[str, WebRuntime] = {}
        self._diagnostic_marks: dict[tuple[str, str], int] = {}

    def _connection_mode(self) -> str:
        configured = self.options.get("connection")
        if configured is None:
            configured = self.config.platform_options(self.name).get("connection", "isolated")
        return str(configured).strip().casefold()

    def supports(self, capability: str) -> bool:
        key = normalize_capability(capability)
        if (
            self._connection_mode() in {"existing-chrome", "existing-cdp"}
            and key in _ATTACHED_UNSAFE_CAPABILITIES
        ):
            return False
        return super().supports(key)

    def validate_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        known = {
            "url",
            "connection",
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
            "attach_timeout_ms",
            "cdp_endpoint",
            "page_url",
            "context_slots",
        }
        unknown = sorted(str(key) for key in options if key not in known)
        if unknown:
            raise ConfigError(
                f"platform 'web' does not accept options: {', '.join(unknown)}",
                hint="See docs/web.md for the supported browser options.",
            )
        normalized: dict[str, Any] = {}
        connection = str(options.get("connection", "isolated")).strip().casefold()
        if connection not in _CONNECTIONS:
            raise ConfigError(
                f"unknown web connection {connection!r}",
                hint="Choose isolated, existing-chrome, or existing-cdp.",
            )
        normalized["connection"] = connection
        if connection == "existing-cdp":
            normalized["cdp_endpoint"] = cdp_endpoint(options.get("cdp_endpoint"))
            if "page_url" in options:
                normalized["page_url"] = cdp_page_url(options["page_url"])
        elif "cdp_endpoint" in options or "page_url" in options:
            raise ConfigError("cdp_endpoint and page_url require connection: existing-cdp")
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
            ("attach_timeout_ms", 30_000),
            ("context_slots", 4),
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
        if connection in {"existing-chrome", "existing-cdp"}:
            incompatible = sorted(
                field
                for field in (
                    "url",
                    "context_slots",
                    "browser",
                    "headless",
                    "channel",
                    "executable_path",
                    "viewport_width",
                    "viewport_height",
                    "navigation_timeout_ms",
                    "storage_state",
                    "ignore_https_errors",
                    "bypass_csp",
                    "service_workers",
                    "proxy_server",
                    "proxy_bypass",
                    "proxy_username",
                    "proxy_password_env",
                )
                if field in options
            )
            if incompatible:
                raise ConfigError(
                    f"{connection} does not accept isolated-browser options: "
                    + ", ".join(incompatible),
                    hint="Use browser lab controls only with connection: isolated.",
                )
        return normalized

    def _launch_options(self) -> WebLaunchOptions:
        values = dict(self.options)
        values.pop("url", None)
        values.pop("connection", None)
        values.pop("attach_timeout_ms", None)
        values.pop("context_slots", None)
        password_env = values.pop("proxy_password_env", None)
        if password_env is not None:
            password = os.environ.get(str(password_env))
            if password is None:
                raise ConfigError(
                    f"web proxy password environment variable {password_env!r} is not set"
                )
            values["proxy_password"] = password
        return WebLaunchOptions(**values)

    def _attach_options(self) -> ChromeAttachOptions:
        return ChromeAttachOptions(
            attach_timeout_ms=int(self.options.get("attach_timeout_ms", 30_000)),
            action_timeout_ms=int(self.options.get("action_timeout_ms", 5_000)),
        )

    def prepare_host(self) -> None:
        uses_playwright = (
            self._uses_default_cdp_launcher if self._connection_mode() == "existing-cdp"
            else self._uses_default_launcher
        )
        if (
            self._connection_mode() != "existing-chrome"
            and uses_playwright
            and importlib.util.find_spec("playwright") is None
        ):
            raise DeviceError(
                "web support needs the optional Playwright dependency",
                code="web_driver_missing",
                hint="Install `android-ui-analyser[web]`, then run `playwright install chromium`.",
            )

    def list_targets(self) -> list[DiscoveredTarget]:
        if self._connection_mode() == "existing-cdp":
            return [
                TargetInfo(
                    target_id=cdp_target_id(str(self.options["cdp_endpoint"])),
                    platform=self.name,
                    status=TargetStatus.online,
                    model="Existing Chromium/Electron",
                    os_name="web",
                )
            ]
        if self._connection_mode() == "existing-chrome":
            return [
                TargetInfo(
                    target_id=ATTACHED_TARGET_ID,
                    platform=self.name,
                    status=TargetStatus.online,
                    model="Chrome extension",
                    os_name="web",
                )
            ]
        candidate = self.options.get("url") or self.config.device.serial
        if not candidate:
            return []
        target = _url(candidate, field="target")
        # A configured URL is a launch destination, not a shared physical browser. Each slot
        # gets an independent warm daemon/context and uses the same owner fencing as devices.
        if self.options.get("url") and not str(self.config.device.serial or "").startswith(
            ("http://", "https://")
        ):
            return [
                TargetInfo(
                    target_id=identity,
                    platform=self.name,
                    status=TargetStatus.online,
                    model=str(self.options.get("browser", "chromium")),
                    os_name="web",
                )
                for identity in self._context_targets(target)
            ]
        return [
            TargetInfo(
                target_id=target,
                platform=self.name,
                status=TargetStatus.online,
                model=str(self.options.get("browser", "chromium")),
                os_name="web",
            )
        ]

    def _context_targets(self, url: str) -> list[str]:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        return [
            f"browser:{key}:{index}" for index in range(int(self.options.get("context_slots", 4)))
        ]

    def probe_target_capabilities(self, target_id: str) -> dict[str, Any]:
        return {
            "headed": self._connection_mode() != "isolated"
            or not self.options.get("headless", True)
        }

    def lease_conflict_hint(self) -> str:
        if self._connection_mode() != "isolated":
            return (
                "This existing browser is exclusive to its owning agent. Let that agent finish "
                "its session, or configure a different browser attachment. Inspect `aua lease list`."
            )
        return (
            "Finish an owned session, or configure platforms.web.url with more context_slots "
            "and omit --serial to claim a separate browser. Inspect `aua lease list`."
        )

    def retain_runtime_on_lease_change(self) -> bool:
        # Isolated runtimes discard the old agent's cookies, controls and buffered events.
        # Attached runtimes only detach their transport, preserving the user's app/profile.
        return False

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        if self._connection_mode() == "existing-cdp":
            endpoint = str(self.options["cdp_endpoint"])
            identity = cdp_target_id(endpoint)
            if (target_id or self.config.device.serial or identity) != identity:
                raise DeviceError(
                    "CDP target does not match the configured endpoint",
                    code="no_target",
                    hint="Omit --serial; the endpoint identifies this target.",
                )
            self.prepare_host()
            cdp_connection = self._cdp_launcher.launch(
                CdpAttachOptions(
                    endpoint=endpoint,
                    page_url=self.options.get("page_url"),
                    attach_timeout_ms=int(self.options["attach_timeout_ms"]),
                    action_timeout_ms=int(self.options["action_timeout_ms"]),
                )
            )
            runtime = WebRuntime(cdp_connection, identity, home_url=cdp_connection.url)
            self._runtimes[identity] = runtime
            return runtime
        if self._connection_mode() == "existing-chrome":
            requested = target_id or self.config.device.serial or ATTACHED_TARGET_ID
            if requested != ATTACHED_TARGET_ID:
                raise DeviceError(
                    f"existing-chrome target must be {ATTACHED_TARGET_ID!r}",
                    code="no_target",
                )
            self.prepare_host()
            connection = self._extension_launcher.launch(ATTACHED_TARGET_ID, self._attach_options())
            runtime = WebRuntime(connection, ATTACHED_TARGET_ID, home_url=connection.url)
            self._runtimes[ATTACHED_TARGET_ID] = runtime
            return runtime
        target = target_id or self.options.get("url")
        if not target:
            raise DeviceError(
                "no web URL configured",
                code="no_target",
                hint="Pass `--serial https://…` or set platforms.web.url in AUA config.",
            )
        identity = str(target)
        if identity.startswith("browser:"):
            configured = self.options.get("url")
            if not configured or identity not in self._context_targets(str(configured)):
                raise DeviceError(
                    "browser context does not belong to this configuration", code="no_target"
                )
            url = str(configured)
        else:
            url = _url(target, field="target")
        self.prepare_host()
        runtime = WebRuntime(
            self._launcher.launch(url, self._launch_options()), identity, home_url=url
        )
        self._runtimes[identity] = runtime
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
        if self._connection_mode() == "existing-cdp":
            installed = importlib.util.find_spec("playwright") is not None
            endpoint = str(self.options["cdp_endpoint"])
            return {
                "platform": {"ok": True, "detail": self.name, "connection": "existing-cdp",
                             "capabilities": sorted(c for c in self.capabilities if self.supports(c))},
                "playwright": {"ok": installed, "detail": "installed" if installed else "not installed"},
                "target": {
                    "ok": self.config.device.serial in {None, cdp_target_id(endpoint)},
                    "detail": endpoint,
                    "hint": "Omit --serial. session start verifies attachment to the running app.",
                },
            }
        if self._connection_mode() == "existing-chrome":
            from ..chrome_extension_setup import chrome_extension_status

            extension = chrome_extension_status()
            attached_configured = self.config.device.serial in {None, ATTACHED_TARGET_ID}
            return {
                "platform": {
                    "ok": True,
                    "detail": self.name,
                    "connection": "existing-chrome",
                    "capabilities": sorted(
                        capability for capability in self.capabilities if self.supports(capability)
                    ),
                },
                "chrome_extension": {
                    "ok": bool(extension["ok"]),
                    "detail": extension["extension_path"],
                    **(
                        {}
                        if extension["ok"]
                        else {"hint": "Run `aua browser extension install`, then load it in Chrome."}
                    ),
                },
                "target": {
                    "ok": attached_configured,
                    "detail": ATTACHED_TARGET_ID,
                    **(
                        {}
                        if attached_configured
                        else {"hint": f"Use --serial {ATTACHED_TARGET_ID} or omit --serial."}
                    ),
                },
            }
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
