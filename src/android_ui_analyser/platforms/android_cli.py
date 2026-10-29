"""Optional Google Android CLI implementation, loaded only by AndroidPlatform.

The core passes semantic operation names and validated values. This module alone knows
the native command grammar. A missing CLI never changes AUA's normal Android backend.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..errors import ConfigError, DevelopmentToolError, UsageError

VERIFIED_VERSION = "1.1.16717390"
MINIMUM_VERSION = (1, 1, 16717390)
OUTPUT_LIMIT = 256 * 1024
_DOWNLOAD_LIMIT = 64 * 1024 * 1024
_CATALOGUE = json.loads(Path(__file__).with_name("android_cli_catalogue.json").read_text())
_OPERATIONS = {item["name"]: item for item in _CATALOGUE["operations"]}
_VERSION = re.compile(r"(?m)^\s*(\d+)\.(\d+)\.(\d+)\s*$")
_RESERVED_OPTIONS = frozenset({"device", "session", "sdk"})
_ARTIFACT_SUFFIXES = frozenset(
    {".png", ".json", ".hprof", ".perfetto-trace", ".apk", ".aab", ".log"}
)
# Verified from the official bundle's InstrumentationServerConnector and APK manifest.
# Layout/input start this persistent instrumentation, which keeps Android's UiAutomation
# slot after the launcher exits. Never guess another package when a CLI release changes it.
_UI_HELPER_PACKAGE = "com.android.cli.interact.instrumentation"
_UI_HELPER_SOCKET = "localabstract:com.android.cli.interact.server"
_UI_HELPER_HASHES = {
    "1.1.16717390": "3949be71bfa29a7f11d6bea8b09b68fa40c2f356d8a6306f0f0988bbc8bb533a",
}


class AndroidCliError(DevelopmentToolError):
    """A native failure retaining bounded, inspectable diagnostic output."""

    def __init__(self, message: str, *, code: str, result: dict[str, Any]) -> None:
        super().__init__(
            message,
            code=code,
            result=result,
            hint="Inspect result.stderr and result.exit_code for the native failure.",
        )


def _managed_binary() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    name = "android.exe" if os.name == "nt" else "android"
    return data_home / "android-ui-analyser" / "tools" / "android-cli" / name


def _read_output(stream: Any) -> tuple[str, bool]:
    size = stream.tell()
    stream.seek(0)
    raw = stream.read(OUTPUT_LIMIT)
    return raw.decode("utf-8", errors="replace"), size > OUTPUT_LIMIT


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """Do not leave a Gradle child writing after the command's deadline."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
    process.wait(timeout=5)


def _run(argv: list[str], *, cwd: str | None, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    # Native builds can print megabytes. Drain to files, retain a bounded result, and
    # never block a child on a full PIPE while waiting for it to finish.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise ConfigError(
                f"Android CLI could not start: {exc}",
                code="android_cli_unavailable",
                hint="Run `aua devtools status` or configure platforms.android.cli_path.",
            ) from exc
        timed_out = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(process)
        except BaseException:
            _kill_tree(process)
            raise
        out, out_cut = _read_output(stdout)
        err, err_cut = _read_output(stderr)
    return {
        "exit_code": process.returncode,
        "stdout": out,
        "stderr": err,
        "stdout_truncated": out_cut,
        "stderr_truncated": err_cut,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "timed_out": timed_out,
    }


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as stream:
        if urlparse(response.url).hostname != "dl.google.com":
            raise ConfigError("Android CLI download redirected outside dl.google.com")
        size = 0
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > _DOWNLOAD_LIMIT:
                raise ConfigError("Android CLI launcher exceeded the download size limit")
            stream.write(chunk)
    prefix = destination.read_bytes()[:4]
    if prefix not in {
        b"\x7fELF",
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xca\xfe\xba\xbe",
    } and not prefix.startswith(b"MZ"):
        raise ConfigError("Android CLI download was not a supported executable")


def _structured_output(output: str) -> tuple[Any, str]:
    """Keep machine output once, separating a native startup banner when present."""
    with contextlib.suppress(ValueError, TypeError):
        return json.loads(output), ""
    decoder = json.JSONDecoder()
    # A launcher may print component-download messages before --format=json. Accept
    # only a complete final document, not a JSON-looking log followed by other output.
    for match in re.finditer(r"(?m)^[ \t]*(?=[{\[])", output):
        with contextlib.suppress(ValueError):
            data, end = decoder.raw_decode(output, match.end())
            if not output[end:].strip():
                return data, output[: match.start()].rstrip()
    return None, output


def _documentation_results(output: str) -> dict[str, Any] | None:
    matches = list(re.finditer(r"(?m)^\d+\. (.+)\n\s+URL: (kb://[^\n]+)\n", output))
    if not matches:
        return None
    results = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        results.append(
            {
                "title": match.group(1),
                "url": match.group(2).strip(),
                "excerpt": output[match.end() : end].strip(),
            }
        )
    return {"results": results}


class AndroidCliTools:
    """Android implementation of the optional development-tools service."""

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = dict(options or {})
        self._version_cache: tuple[str, int, int, dict[str, Any]] | None = None

    def _binary(self) -> Path | None:
        explicit = self.options.get("cli_path") or os.environ.get("AUA_ANDROID_CLI")
        if explicit:
            path = Path(str(explicit)).expanduser()
            if not path.is_absolute():
                found = shutil.which(str(path))
                if found:
                    path = Path(found)
            return path.resolve() if path.is_file() and os.access(path, os.X_OK) else None
        managed = _managed_binary()
        if managed.is_file() and os.access(managed, os.X_OK):
            return managed
        found = shutil.which("android")
        return Path(found) if found else None

    def status(self) -> dict[str, Any]:
        binary = self._binary()
        result: dict[str, Any] = {
            "available": binary is not None,
            "compatible": False,
            "binary": str(binary) if binary else None,
            "version": None,
            "verified_version": VERIFIED_VERSION,
            "optional": True,
            "install_hint": "aua devtools setup",
            "download_url": "https://developer.android.com/tools/agents/android-cli/download",
        }
        if binary is None:
            return result
        stat = binary.stat()
        key = (str(binary), stat.st_mtime_ns, stat.st_size)
        if self._version_cache and self._version_cache[:3] == key:
            return copy.deepcopy(self._version_cache[3])
        try:
            probe = _run([str(binary), "--no-metrics", "--version"], cwd=None, timeout_s=20)
            match = _VERSION.search(probe["stdout"]) or _VERSION.search(probe["stderr"])
            if match and probe["exit_code"] == 0 and not probe["timed_out"]:
                version = tuple(int(part) for part in match.groups())
                result.update(
                    version=".".join(match.groups()), compatible=version >= MINIMUM_VERSION
                )
            if not result["compatible"]:
                result["reason"] = (
                    "AUA requires Google Android CLI 1.1.16717390 or later; the legacy SDK android command is not supported."
                )
        except ConfigError as exc:
            result["reason"] = exc.message
        self._version_cache = (*key, copy.deepcopy(result))
        return result

    def _require_binary(self) -> str:
        status = self.status()
        if not status["available"]:
            raise ConfigError(
                "Optional Android CLI is not installed or its configured path is unavailable",
                code="android_cli_missing",
                hint="Run `aua devtools setup`, install Google Android CLI yourself, or set platforms.android.cli_path.",
            )
        if not status["compatible"]:
            raise ConfigError(
                str(status.get("reason", "Unsupported Android CLI version")),
                code="android_cli_unsupported_version",
            )
        return str(status["binary"])

    def require_available(self) -> str:
        """Check optional-tool readiness before a runtime releases its UI connection."""
        return self._require_binary()

    def setup(self) -> dict[str, Any]:
        """Explicit per-user install from Google; no sudo, shell script, or PATH edits."""
        existing = self.status()
        if existing["compatible"]:
            return {"ok": True, "installed": False, **existing}
        machine = platform.machine().lower()
        arch = (
            "arm64"
            if machine in {"arm64", "aarch64"}
            else "x86_64"
            if machine in {"amd64", "x86_64"}
            else None
        )
        system = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(platform.system())
        if system is None or arch is None or (system == "windows" and arch != "x86_64"):
            raise ConfigError(
                "No Android CLI installer is configured for this host",
                code="android_cli_host_unsupported",
            )
        destination = _managed_binary()
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://dl.google.com/android/cli/latest/{system}_{arch}/{destination.name}"
        fd, temporary = tempfile.mkstemp(prefix=".android-download-", dir=destination.parent)
        os.close(fd)
        path = Path(temporary)
        try:
            _download(url, path)
            path.chmod(0o700)
            probe = _run([str(path), "--no-metrics", "--version"], cwd=None, timeout_s=600)
            match = _VERSION.search(probe["stdout"]) or _VERSION.search(probe["stderr"])
            if (
                probe["exit_code"] != 0
                or probe["timed_out"]
                or not match
                or tuple(int(x) for x in match.groups()) < MINIMUM_VERSION
            ):
                raise ConfigError(
                    "Downloaded Android CLI failed its version check",
                    code="android_cli_setup_failed",
                )
            path.replace(destination)
        finally:
            path.unlink(missing_ok=True)
        self._version_cache = None
        installed = AndroidCliTools({"cli_path": str(destination)}).status()
        result = {"ok": installed["compatible"], "installed": True, **installed}
        if self.options.get("cli_path") or os.environ.get("AUA_ANDROID_CLI"):
            result["configuration_hint"] = (
                f"Set platforms.android.cli_path to {destination} or remove the explicit override."
            )
        return result

    def catalogue(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(_OPERATIONS.values()))

    def validate(
        self,
        operation: str,
        arguments: list[str],
        options: dict[str, Any],
        project_dir: str | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        spec = _OPERATIONS.get(operation)
        if spec is None:
            raise UsageError(
                f"Unknown development-tool operation: {operation}",
                hint="Use `aua devtools catalogue` to discover operation names.",
            )
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 3600
        ):
            raise UsageError("timeout_s must be a finite number between 0 and 3600")
        if not isinstance(arguments, list) or any(
            not isinstance(arg, str) or "\x00" in arg or arg.startswith(("-", "@"))
            for arg in arguments
        ):
            raise UsageError(
                "arguments must contain positional strings, not flags or response files"
            )
        bounds = spec["arguments"]
        if not bounds["minimum"] <= len(arguments) <= bounds["maximum"]:
            raise UsageError(
                f"{operation} expects {bounds['minimum']}..{bounds['maximum']} positional arguments"
            )
        if not isinstance(options, dict) or any(not isinstance(key, str) for key in options):
            raise UsageError("options must be an object with named keys")
        reserved = _RESERVED_OPTIONS.intersection(options)
        if reserved:
            raise UsageError(
                f"AUA owns these options: {', '.join(sorted(reserved))}",
                hint="Select the leased target through AUA; profiling sessions are assigned by AUA.",
            )
        unknown = set(options) - set(spec["options"])
        if unknown:
            raise UsageError(f"Unsupported {operation} options: {', '.join(sorted(unknown))}")
        for key, schema in spec["options"].items():
            if schema.get("required") and (key not in options or options[key] in (None, "", [])):
                raise UsageError(f"{operation} requires option {key}")
            if key not in options:
                continue
            value = options[key]
            if schema["type"] == "boolean":
                if not isinstance(value, bool):
                    raise UsageError(f"{key} must be a boolean")
                continue
            if schema["type"] == "array":
                if not isinstance(value, list) or any(
                    not isinstance(item, str) or "\x00" in item or item.startswith("@")
                    for item in value
                ):
                    raise UsageError(f"{key} must be an array of strings without response files")
                continue
            values = value if isinstance(value, list) and schema.get("repeatable") else [value]
            if not values:
                raise UsageError(f"{key} must not be empty")
            for item in values:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (str, int, float))
                    or "\x00" in str(item)
                    or str(item) == ""
                    or (isinstance(item, float) and not math.isfinite(item))
                ):
                    raise UsageError(f"{key} requires a string or finite number")
                if schema.get("enum") and item not in schema["enum"]:
                    raise UsageError(f"{key} must be one of {schema['enum']}")
        if project_dir is not None and (
            not isinstance(project_dir, str) or not Path(project_dir).expanduser().is_dir()
        ):
            raise UsageError("project_dir must name an existing directory")
        if operation == "update" and options.get("url"):
            url = urlparse(str(options["url"]))
            if (
                url.scheme != "https"
                or url.hostname != "dl.google.com"
                or not url.path.startswith("/android/cli/")
            ):
                raise UsageError("Android CLI updates must use the official HTTPS download origin")
        if operation == "profiler.heapdump.capture" and (
            ("pid" in options) == ("process" in options)
            or not (options.get("pid") or options.get("process"))
        ):
            raise UsageError("heap capture requires exactly one of process or pid")
        result = copy.deepcopy(spec)
        if (
            operation == "create"
            and options.get("list")
            or operation == "emulator.create"
            and options.get("list-profiles")
        ):
            result.update(effect="read", needs_confirmation=False)
        if options.get("force"):
            result["needs_confirmation"] = True
        if (
            operation == "profiler.systemtrace.start"
            and options.get("duration")
            and options.get("wait")
        ):
            result["next_operation"] = None
        return result

    def execute(
        self,
        operation: str,
        arguments: list[str],
        options: dict[str, Any],
        project_dir: str | None,
        timeout_s: float,
        target_id: str | None = None,
    ) -> dict[str, Any]:
        options = dict(options)
        session = options.pop("session", None)
        spec = self.validate(operation, arguments, options, project_dir, timeout_s)
        if spec["route"]:
            raise UsageError(f"{operation} must use AUA's owned virtual-target route")
        if spec["scope"] == "target" and not target_id:
            raise UsageError(f"{operation} requires an explicit leased target")
        if spec["scope"] == "host" and target_id is not None:
            raise UsageError(f"{operation} is a host operation; no target may be supplied")
        if spec["profile_action"]:
            if not isinstance(session, str) or not re.fullmatch(
                r"aua-[A-Za-z0-9_-]{8,96}", session
            ):
                raise UsageError("Profiling requires an AUA-owned explicit session ID")
        elif session is not None:
            raise UsageError("This operation does not accept a managed profiling session")
        binary = self._require_binary()
        command = operation.split(".")
        argv = [binary, "--no-metrics"]
        if self.options.get("cli_sdk"):
            argv.append(f"--sdk={Path(str(self.options['cli_sdk'])).expanduser()}")
        argv.extend(command)
        if target_id is not None:
            if not isinstance(target_id, str) or not target_id or "\x00" in target_id:
                raise UsageError("Invalid target identity")
            argv.append(f"--device={target_id}")
        if session is not None:
            argv.append(f"--session={session}")
        if "format" in spec["options"] and "format" not in options:
            options["format"] = "json"
        build_args = options.pop("build-args", [])
        for key, value in options.items():
            native_key = "device" if key == "device-filter" else key
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, bool):
                    encoded = str(item).lower()
                elif isinstance(item, float) and item.is_integer():
                    encoded = str(int(item))
                else:
                    encoded = str(item)
                argv.append(f"--{native_key}={encoded}")
        argv.extend(arguments)
        if build_args:
            argv.extend(["--", *build_args])
        cwd = str(Path(project_dir).expanduser().resolve()) if project_dir else None
        result = _run(argv, cwd=cwd, timeout_s=timeout_s)
        result.update(
            operation=operation,
            provider="android-cli",
            target_id=target_id,
            ok=result["exit_code"] == 0 and not result["timed_out"],
        )
        if session is not None:
            result["session_id"] = session
        result["data"] = None
        if not result["stdout_truncated"]:
            result["data"], remaining = _structured_output(result["stdout"])
            if result["ok"]:
                result["stdout"] = remaining
            if result["ok"] and operation == "docs.search" and result["data"] is None:
                parsed_docs = _documentation_results(result["stdout"])
                if parsed_docs is not None:
                    result["data"] = parsed_docs
                    result["stdout"] = ""
        artifact_data = [result["data"]]
        artifact_data.extend(
            match.group(1).strip().strip('"')
            for match in re.finditer(r"(?m)^Build log:\s*(.+)$", result["stdout"])
        )
        if operation.startswith("profiler.") and result["ok"]:
            # Native guidance bypasses AUA's lease/session boundary. Return semantic
            # next-operation metadata for the core to turn into an owned AUA call.
            next_match = re.search(
                r"(?m)^Next: android profiler systemtrace (start|stop)\b.*$", result["stdout"]
            )
            if next_match:
                result["next_operation"] = "profiler.systemtrace." + next_match.group(1)
            result["stdout"] = re.sub(r"(?m)^Next: android [^\n]*(?:\n|$)", "", result["stdout"])
            artifact_data.extend(
                match.group(1).strip().strip('"')
                for match in re.finditer(
                    r"(?im)^(?:Trace|Heap dump|Heap snapshot) saved (?:at|to):\s*(.+)$",
                    result["stdout"],
                )
            )
        result["artifacts"] = self._artifacts(artifact_data, options, cwd)
        if spec.get("retained_resources"):
            result["retained_resources"] = spec["retained_resources"]
        if result["timed_out"]:
            result["outcome_unknown"] = spec["scope"] == "target" and spec["effect"] != "read"
            if result["outcome_unknown"]:
                result["next_action"] = (
                    "Inspect the target before continuing; do not repeat the operation."
                )
            raise AndroidCliError(
                "Android CLI exceeded the operation deadline",
                code="android_cli_timeout",
                result=result,
            )
        if not result["ok"]:
            # Native failure may occur after installation, input delivery or profiler
            # setup. An exit status alone is not proof the leased target was untouched.
            result["outcome_unknown"] = spec["scope"] == "target" and spec["effect"] != "read"
            if result["outcome_unknown"]:
                result["next_action"] = (
                    "Inspect the target before continuing; do not repeat the operation."
                )
            raise AndroidCliError(
                f"Android CLI {operation} exited with status {result['exit_code']}",
                code="android_cli_failed",
                result=result,
            )
        if operation == "update":
            self._version_cache = None
        return result

    @staticmethod
    def _artifacts(data: Any, options: dict[str, Any], cwd: str | None) -> list[dict[str, Any]]:
        candidates: set[str] = set()
        pending = [data]
        inspected = 0
        while pending and inspected < 5000:
            item = pending.pop()
            inspected += 1
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
            elif (
                isinstance(item, str)
                and len(item) < 4096
                and Path(item).suffix.lower() in _ARTIFACT_SUFFIXES
            ):
                candidates.add(item)
        for key in ("output", "output-image-file"):
            if isinstance(options.get(key), str):
                candidates.add(options[key])
        artifacts = []
        for candidate in sorted(candidates):
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = Path(cwd or os.getcwd()) / path
            try:
                if path.is_file():
                    artifacts.append(
                        {
                            "path": str(path.resolve()),
                            "size_bytes": path.stat().st_size,
                            "kind": path.suffix.lstrip("."),
                        }
                    )
            except (OSError, ValueError):
                continue
            if len(artifacts) >= 100:
                break
        return artifacts

    def terminate_profile(self, session_id: str, target_id: str) -> dict[str, Any]:
        """Restore only the caller's exact session. Never invoke implicit 'the only session'."""
        try:
            return self.execute(
                "profiler.session.terminate", [], {"session": session_id}, None, 30, target_id
            )
        except AndroidCliError as exc:
            # A missing exact session is already clean. Other failures retain the ledger
            # entry so another reaper can retry safely.
            text = (exc.result.get("stdout", "") + "\n" + exc.result.get("stderr", "")).casefold()
            missing_exact = re.search(
                rf"\bsession ['\"]?{re.escape(session_id.casefold())}['\"]? (?:was )?not found\b",
                text,
            )
            if not exc.result.get("timed_out") and (
                missing_exact
                or any(
                    marker in text
                    for marker in ("no session found", "session not found", "no active session")
                )
            ):
                return {"ok": True, "already_finished": True, "session_id": session_id}
            raise

    def _helper_command(self, target_id: str, arguments: list[str]) -> dict[str, Any]:
        if not isinstance(target_id, str) or not target_id or "\x00" in target_id:
            raise UsageError("Helper cleanup requires an explicit target identity")
        sdk = (
            self.options.get("cli_sdk")
            or os.environ.get("ANDROID_HOME")
            or os.environ.get("ANDROID_SDK_ROOT")
        )
        if sdk:
            adb = (
                Path(str(sdk)).expanduser()
                / "platform-tools"
                / ("adb.exe" if os.name == "nt" else "adb")
            )
            executable = str(adb) if adb.is_file() else None
        else:
            executable = shutil.which("adb")
        if not executable:
            raise ConfigError("ADB is required to restore Android CLI's UI helper")
        result = _run([executable, "-s", target_id, *arguments], cwd=None, timeout_s=15)
        result.update(operation="ui-helper", target_id=target_id, provider="android-cli")
        output = result["stdout"] + "\n" + result["stderr"]
        if (
            result["exit_code"] != 0
            or result["timed_out"]
            or result["stdout_truncated"]
            or re.search(r"(?im)^\s*(?:error:|failure\b|.*exception:)", output)
        ):
            raise AndroidCliError(
                "Android CLI UI helper preparation or restoration failed",
                code="android_cli_failed",
                result=result,
            )
        return result

    def _ui_helper_installed(self, target_id: str) -> bool:
        result = self._helper_command(
            target_id, ["shell", "pm", "list", "packages", _UI_HELPER_PACKAGE]
        )
        return f"package:{_UI_HELPER_PACKAGE}" in result["stdout"].splitlines()

    def _ui_helper_instance(self, target_id: str) -> str:
        token = self._helper_command(
            target_id, ["shell", "cat", "/proc/sys/kernel/random/boot_id"]
        )["stdout"].strip()
        if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", token):
            raise ConfigError("Cannot identify the Android CLI helper's target boot")
        return token

    def _ui_helper_forwards(self, target_id: str) -> list[str]:
        result = self._helper_command(target_id, ["forward", "--list"])
        ports = []
        for line in result["stdout"].splitlines():
            fields = line.split()
            if (
                len(fields) == 3
                and fields[0] == target_id
                and fields[2] == _UI_HELPER_SOCKET
                and re.fullmatch(r"tcp:[0-9]+", fields[1])
            ):
                ports.append(fields[1])
        return sorted(set(ports))

    def prepare_ui_helper(self, target_id: str) -> dict[str, Any]:
        """Read the baseline before the core records the durable helper undo."""
        self.require_available()
        version = self.status()["version"]
        expected_hash = _UI_HELPER_HASHES.get(version)
        if expected_hash is None:
            raise ConfigError(
                f"Android CLI {version} UI helper lifecycle has not been verified",
                code="android_cli_unsupported_version",
                hint=f"Use Android CLI {VERIFIED_VERSION} for layout/input operations.",
            )
        instance_token = self._ui_helper_instance(target_id)
        forwards = self._ui_helper_forwards(target_id)
        installed = self._ui_helper_installed(target_id)
        if installed:
            result = self._helper_command(target_id, ["shell", "pm", "path", _UI_HELPER_PACKAGE])
            paths = [
                line[8:] for line in result["stdout"].splitlines() if line.startswith("package:")
            ]
            if len(paths) != 1 or not paths[0].startswith("/") or "\x00" in paths[0]:
                raise ConfigError("Cannot verify the preexisting Android CLI UI helper APK")
            digest = self._helper_command(
                target_id, ["shell", "sha256sum " + shlex.quote(paths[0])]
            )["stdout"].split()
            if not digest or digest[0] != expected_hash:
                raise ConfigError(
                    "A different Android CLI UI helper is already installed on this target",
                    code="android_cli_helper_conflict",
                    hint="Explicitly upgrade or remove that helper before using AUA layout/input; "
                    "AUA will not overwrite a preexisting helper APK.",
                )
        return {
            "remove_installed_helper": not installed,
            "instance_token": instance_token,
            "forward_ports": forwards,
        }

    def release_ui_helper(self, target_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """Replay an AUA-owned helper undo without needing a working UI connection.

        The core records this state before native execution and retains it on failure.
        Only an APK absent from the baseline is removed; a matching preexisting APK stays.
        """
        if not isinstance(state, dict) or type(state.get("remove_installed_helper")) is not bool:
            raise UsageError("Invalid Android CLI UI helper restoration state")
        if not state.get("instance_token"):
            raise ConfigError("Android CLI UI helper restoration has no recorded boot identity")
        baseline_forwards = state.get("forward_ports")
        if not isinstance(baseline_forwards, list) or any(
            not isinstance(port, str) or not re.fullmatch(r"tcp:[0-9]+", port)
            for port in baseline_forwards
        ):
            raise ConfigError("Android CLI UI helper restoration has no valid forwarding baseline")
        if self._ui_helper_instance(target_id) != state["instance_token"]:
            raise ConfigError(
                "Android CLI UI helper belongs to a different target boot",
                code="android_cli_helper_instance_changed",
            )
        self._helper_command(target_id, ["shell", "am", "force-stop", _UI_HELPER_PACKAGE])
        new_forwards = set(self._ui_helper_forwards(target_id)) - set(baseline_forwards)
        if len(new_forwards) > 32:
            raise ConfigError("Too many new Android CLI helper forwards to restore safely")
        for port in sorted(new_forwards):
            self._helper_command(target_id, ["forward", "--remove", port])
        removed = False
        if state["remove_installed_helper"] and self._ui_helper_installed(target_id):
            self._helper_command(target_id, ["uninstall", _UI_HELPER_PACKAGE])
            if self._ui_helper_installed(target_id):
                raise ConfigError("Android CLI UI helper remained installed after its undo")
            removed = True
        return {
            "ok": True,
            "helper_stopped": True,
            "helper_removed": removed,
            "forwards_removed": sorted(new_forwards),
        }
