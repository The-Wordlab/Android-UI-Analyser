"""Private transport for selected-platform options across detached process boundaries.

Platform options can include endpoints and accidentally pasted credentials, so they must not be
put in argv (visible in process listings) or synthesized into environment variables. Detached
workers inherit one anonymous file descriptor containing only the selected platform's JSON
mapping. The child replaces, rather than merges, any options discovered from its own cwd or user
configuration.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import UsageError

if TYPE_CHECKING:
    from ..config import Config


PLATFORM_ENV_PREFIX = "AUA_PLATFORMS__"
MAX_PLATFORM_OPTIONS_BYTES = 1_048_576


def selected_platform_options(
    config: Config,
    platform: str,
    *,
    mask_secrets: bool = False,
) -> dict[str, Any]:
    """Return one plugin's JSON-compatible effective option mapping."""

    if not mask_secrets:
        return config.platform_options(platform)
    data = config.masked_dict()
    platforms = data.get("platforms")
    if not isinstance(platforms, dict):  # pragma: no cover - Config guarantees this shape
        return {}
    options = platforms.get(platform.strip().lower(), {})
    return dict(options) if isinstance(options, dict) else {}


def encode_platform_options(options: Mapping[str, Any]) -> bytes:
    """Canonical representation written to an inherited anonymous descriptor."""

    return json.dumps(
        dict(options),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def selected_platform_options_payload(config: Config, platform: str) -> bytes:
    return encode_platform_options(selected_platform_options(config, platform))


def _fingerprint_key(key_dir: str | os.PathLike[str]) -> bytes:
    """Return a local 0600 HMAC key shared by parent and detached children."""

    directory = Path(key_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".platform-options-hmac-key"
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise UsageError("could not read the local platform-option identity key") from exc
    else:
        if len(existing) < 32:
            raise UsageError("the local platform-option identity key is invalid")
        with contextlib.suppress(OSError):
            path.chmod(0o600)
        return existing
    temporary = directory / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    fd: int | None = None
    try:
        # Publish a fully written inode with one atomic link. Creating the final path before
        # writing lets a simultaneous first caller observe an empty/partial key and fail even
        # though both processes are correctly configured.
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        payload = secrets.token_bytes(32)
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:  # pragma: no cover - defensive kernel/filesystem failure
                raise OSError("short write while creating platform-option identity key")
            written += count
        os.fsync(fd)
        os.close(fd)
        fd = None
        with contextlib.suppress(FileExistsError):
            os.link(temporary, path)
    finally:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(OSError):
            temporary.unlink()
    try:
        key = path.read_bytes()
    except OSError as exc:
        raise UsageError("could not read the local platform-option identity key") from exc
    if len(key) < 32:
        raise UsageError("the local platform-option identity key is invalid")
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return key


def platform_options_fingerprint(
    options: Mapping[str, Any], *, key_dir: str | os.PathLike[str]
) -> str:
    """Keyed identity for the full opaque mapping, safe to persist and compare."""

    if not options:
        # An empty mapping contains no secret or routing choice to hide. Default Android
        # must not become unrecoverable just because a cache-pruner removed a local key.
        return hashlib.sha256(b"aua-empty-platform-options-v1").hexdigest()
    referenced: dict[str, str | None] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower()
                if snake.endswith("_env") and isinstance(item, str) and item:
                    referenced[item] = os.environ.get(item)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(options)
    identity = {"options": dict(options), "referenced_environment": referenced}
    payload = encode_platform_options(identity)
    return hmac.new(_fingerprint_key(key_dir), payload, hashlib.sha256).hexdigest()


def read_platform_options_fd(fd: int, *, consumer: str) -> dict[str, Any]:
    """Read and validate an inherited selected-platform mapping once."""

    try:
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(MAX_PLATFORM_OPTIONS_BYTES + 1)
        if len(raw) > MAX_PLATFORM_OPTIONS_BYTES:
            raise ValueError("payload is too large")
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise UsageError(f"invalid {consumer} platform-option transport") from exc
    if not isinstance(value, dict):
        raise UsageError(f"{consumer} platform options must be a JSON object")
    return value


def scrub_platform_option_environment() -> dict[str, str]:
    """Inherit ordinary environment state but remove deep-merge option overrides."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(PLATFORM_ENV_PREFIX)
    }


__all__ = [
    "MAX_PLATFORM_OPTIONS_BYTES",
    "PLATFORM_ENV_PREFIX",
    "encode_platform_options",
    "platform_options_fingerprint",
    "read_platform_options_fd",
    "scrub_platform_option_environment",
    "selected_platform_options",
    "selected_platform_options_payload",
]
