"""Platform-neutral browser lab controls shared by CLI and MCP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Engine


def _runtime(self: Engine, capability: str) -> Any:
    return self.platform.runtime_capability(capability, self.device)


def browser_status(self: Engine) -> dict[str, Any]:
    """Return the active browser context, pages, and network controls."""

    context = _runtime(self, "ui.tree").current_app()
    capabilities = {
        name: self.platform.supports(name)
        for name in (
            "browser.storage",
            "browser.network",
            "browser.diagnostics",
            "browser.pages",
            "browser.trace",
            "session.state",
        )
    }
    result: dict[str, Any] = {
        "ok": True,
        "action": "browser-status",
        "target_id": self.device.target_id,
        "context": context.model_dump(mode="json"),
        "capabilities": capabilities,
    }
    if capabilities["browser.network"]:
        result["network"] = _runtime(self, "browser.network").browser_network_status()
    if capabilities["browser.pages"]:
        result["pages"] = _runtime(self, "browser.pages").browser_pages().get("pages", [])
    return result


def browser_logs(
    self: Engine,
    *,
    limit: int = 100,
    kinds: Sequence[str] = (),
    since_ms: int | None = None,
) -> dict[str, Any]:
    return _runtime(self, "browser.diagnostics").browser_diagnostics(
        limit=limit, kinds=kinds, since_ms=since_ms
    )


def browser_logs_clear(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.diagnostics").browser_diagnostics_clear()


def browser_storage(
    self: Engine, *, include_values: bool = False
) -> dict[str, Any]:
    return _runtime(self, "browser.storage").browser_storage(
        include_values=include_values
    )


def browser_storage_export(self: Engine, path: str) -> dict[str, Any]:
    return _runtime(self, "browser.storage").browser_storage_export(path)


def browser_storage_import(self: Engine, path: str) -> dict[str, Any]:
    return _runtime(self, "browser.storage").browser_storage_import(path)


def browser_storage_clear(
    self: Engine, kinds: Sequence[str] = ()
) -> dict[str, Any]:
    return _runtime(self, "browser.storage").browser_storage_clear(kinds)


def browser_cache_clear(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.storage").browser_cache_clear()


def browser_reset(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.storage").browser_reset()


def browser_network_status(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_network_status()


def browser_offline(self: Engine, offline: bool = True) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_set_offline(offline)


def browser_throttle(
    self: Engine,
    *,
    latency_ms: int = 0,
    download_kbps: int = 0,
    upload_kbps: int = 0,
) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_set_throttle(
        latency_ms=latency_ms,
        download_kbps=download_kbps,
        upload_kbps=upload_kbps,
    )


def browser_cors_add(
    self: Engine,
    *,
    origin: str,
    hosts: Sequence[str],
    methods: Sequence[str] = (),
    headers: Sequence[str] = (),
    credentials: bool = False,
) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_set_cors(
        origin=origin,
        hosts=hosts,
        methods=methods,
        headers=headers,
        credentials=credentials,
    )


def browser_cors_clear(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_clear_cors()


def browser_proxy_set(
    self: Engine,
    server: str,
    *,
    bypass: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_set_proxy(
        server,
        bypass=bypass,
        username=username,
        password=password,
    )


def browser_proxy_clear(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_clear_proxy()


def browser_har_start(self: Engine, path: str) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_har_start(path)


def browser_har_stop(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_har_stop()


def browser_har_replay(
    self: Engine,
    path: str,
    *,
    url: str | None = None,
    not_found: str = "abort",
) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_har_replay(
        path, url=url, not_found=not_found
    )


def browser_har_clear(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_har_clear()


def browser_mock_add(
    self: Engine,
    url: str,
    *,
    status: int = 200,
    body: str = "",
    headers: Mapping[str, str] | None = None,
    abort: bool = False,
) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_mock_add(
        url,
        status=status,
        body=body,
        headers=headers,
        abort=abort,
    )


def browser_mock_clear(
    self: Engine, rule_id: str | None = None
) -> dict[str, Any]:
    return _runtime(self, "browser.network").browser_mock_clear(rule_id)


def browser_pages(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.pages").browser_pages()


def browser_page_select(self: Engine, page_id: str) -> dict[str, Any]:
    return _runtime(self, "browser.pages").browser_page_select(page_id)


def browser_page_close(self: Engine, page_id: str) -> dict[str, Any]:
    return _runtime(self, "browser.pages").browser_page_close(page_id)


def browser_trace_start(self: Engine) -> dict[str, Any]:
    return _runtime(self, "browser.trace").browser_trace_start()


def browser_trace_stop(self: Engine, path: str) -> dict[str, Any]:
    return _runtime(self, "browser.trace").browser_trace_stop(path)


__all__ = [name for name in globals() if name.startswith("browser_")]
