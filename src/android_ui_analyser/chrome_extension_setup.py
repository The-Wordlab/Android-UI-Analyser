"""Install and inspect the bundled Chrome extension/native-host integration."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .platforms.chrome_extension import BRIDGE_HOST_NAME

EXTENSION_ID = "bjbfhnhjmddoehdkhjjnlpjfepbfkncb"


def _data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def extension_install_dir() -> Path:
    return _data_home() / "android-ui-analyser" / "chrome-extension"


def host_wrapper_path() -> Path:
    return _data_home() / "android-ui-analyser" / "chrome-host" / "aua-chrome-host"


def native_manifest_path(browser: str = "chrome") -> Path:
    candidate = browser.strip().casefold()
    if sys.platform == "darwin":
        roots = {
            "chrome": Path.home()
            / "Library/Application Support/Google/Chrome/NativeMessagingHosts",
            "chromium": Path.home()
            / "Library/Application Support/Chromium/NativeMessagingHosts",
            "chrome-beta": Path.home()
            / "Library/Application Support/Google/Chrome Beta/NativeMessagingHosts",
            "chrome-canary": Path.home()
            / "Library/Application Support/Google/Chrome Canary/NativeMessagingHosts",
            "chrome-for-testing": Path.home()
            / "Library/Application Support/Google/ChromeForTesting/NativeMessagingHosts",
        }
    elif sys.platform.startswith("linux"):
        roots = {
            "chrome": Path.home() / ".config/google-chrome/NativeMessagingHosts",
            "chromium": Path.home() / ".config/chromium/NativeMessagingHosts",
            "chrome-beta": Path.home() / ".config/google-chrome-beta/NativeMessagingHosts",
            "chrome-canary": Path.home() / ".config/google-chrome-canary/NativeMessagingHosts",
            "chrome-for-testing": Path.home()
            / ".config/google-chrome-for-testing/NativeMessagingHosts",
        }
    else:
        raise ConfigError(
            "AUA Chrome extension setup currently supports macOS and Linux",
            hint="The isolated Playwright web mode remains available on this operating system.",
        )
    if candidate not in roots:
        raise ConfigError(
            f"unknown Chrome installation {browser!r}",
            hint=(
                "Choose chrome, chromium, chrome-beta, chrome-canary, or chrome-for-testing."
            ),
        )
    return roots[candidate] / f"{BRIDGE_HOST_NAME}.json"


def install_chrome_extension(browser: str = "chrome") -> dict[str, Any]:
    destination = extension_install_dir()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_ref = files("android_ui_analyser").joinpath("chrome_extension")
    with as_file(source_ref) as source:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    wrapper = host_wrapper_path()
    wrapper.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Keep a virtualenv/uv-tool interpreter path intact. Resolving its symlink selects the base
    # Python and drops the environment that actually contains android_ui_analyser.
    executable = str(Path(sys.executable).absolute()).replace("'", "'\\''")
    wrapper.write_text(
        "#!/bin/sh\n" + f"exec '{executable}' -m android_ui_analyser.chrome_native_host\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    manifest = native_manifest_path(browser)
    manifest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest.write_text(
        json.dumps(
            {
                "name": BRIDGE_HOST_NAME,
                "description": "AUA bridge for one user-approved Chrome tab",
                "path": str(wrapper),
                "type": "stdio",
                "allowed_origins": [f"chrome-extension://{EXTENSION_ID}/"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    return {
        "ok": True,
        "action": "browser-extension-install",
        "browser": browser,
        "extension_id": EXTENSION_ID,
        "extension_path": str(destination),
        "native_host_manifest": str(manifest),
        "next": [
            "Open chrome://extensions",
            "Enable Developer mode",
            f"Choose Load unpacked and select {destination}",
            "Open AUA on the tab you want to share and choose Attach this tab",
        ],
    }


def chrome_extension_status(browser: str = "chrome") -> dict[str, Any]:
    extension = extension_install_dir()
    manifest = native_manifest_path(browser)
    wrapper = host_wrapper_path()
    installed = (extension / "manifest.json").is_file() and manifest.is_file() and wrapper.is_file()
    return {
        "ok": installed,
        "action": "browser-extension-status",
        "browser": browser,
        "extension_id": EXTENSION_ID,
        "extension_path": str(extension),
        "extension_files": (extension / "manifest.json").is_file(),
        "native_host_manifest": str(manifest),
        "native_host_registered": manifest.is_file(),
        "native_host_executable": wrapper.is_file() and os.access(wrapper, os.X_OK),
    }


__all__ = [
    "EXTENSION_ID",
    "chrome_extension_status",
    "extension_install_dir",
    "host_wrapper_path",
    "install_chrome_extension",
    "native_manifest_path",
]
