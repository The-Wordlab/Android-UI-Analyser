"""Inject host PCM audio into an Android Emulator microphone.

The emulator exposes ``injectAudio`` as an authenticated client-streaming gRPC.
This module deliberately keeps that optional boundary small: ``grpcio`` is imported
only when an injection is prepared, while the protobuf messages are encoded here.
That avoids shipping generated emulator bindings (and a runtime ``grpc_tools``
dependency) for two tiny messages whose wire format is stable.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import shlex
import subprocess
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

from .errors import AuaError, DeviceError, UsageError

_INJECT_AUDIO_METHOD = "/android.emulation.control.EmulatorController/injectAudio"
_MAX_SAMPLE_RATE = 48_000
_CHUNK_MS = 100
_AFFECTED_SINGLE_INJECTION_VERSION = (36, 4, 10)
MAX_WAV_DURATION_S = 300.0
SPEECH_SYNTHESIS_TIMEOUT_S = 120.0


class MicDeliveryUncertainError(DeviceError):
    """The stream ended ambiguously after the emulator may have consumed every packet."""

    code = "mic_delivery_uncertain"

    def __init__(
        self,
        message: str | None = None,
        *,
        hint: str | None = None,
        result: dict[str, Any] | None = None,
        followup_errors: Sequence[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            message
            or (
                "the emulator ended the audio stream with INTERNAL after accepting packets; "
                "the samples may already have been delivered"
            ),
            code=self.code,
            hint=hint
            or (
                "Do not retry blindly. Inspect error.result.observation (or the current UI) "
                "for the audio effect; restart only this emulator if it went offline."
            ),
        )
        self.result = result
        self.followup_errors = [dict(error) for error in (followup_errors or ())]

    def with_result(self, result: dict[str, Any]) -> Self:
        self.result = result
        return self

    def note_followup_failure(self, stage: str, exc: BaseException) -> Self:
        """Retain cleanup/observation failures without hiding the no-retry outcome."""

        if isinstance(exc, AuaError):
            code = exc.code
            message = exc.message
        else:
            code = type(exc).__name__
            message = f"{stage.replace('_', ' ')} failed after ambiguous audio delivery"
        self.followup_errors.append({"stage": stage, "code": code, "message": message})
        note = (
            f"The follow-up {stage.replace('_', ' ')} also failed ({code}); "
            "verify the current device state manually."
        )
        if self.hint and note not in self.hint:
            self.hint = f"{self.hint} {note}"
        return self

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        error = payload.get("error")
        if self.result is not None and isinstance(error, dict):
            error["result"] = self.result
        if self.followup_errors and isinstance(error, dict):
            error["followup_errors"] = list(self.followup_errors)
        return payload


class MicDeliveredReleaseError(MicDeliveryUncertainError):
    """Audio completed, but the optional hold could not be released cleanly."""

    code = "mic_delivered_release_failed"

    def __init__(
        self,
        message: str | None = None,
        *,
        hint: str | None = None,
        result: dict[str, Any] | None = None,
        followup_errors: Sequence[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            message
            or (
                "the microphone audio was delivered, but AUA could not release the held "
                "control cleanly"
            ),
            hint=hint
            or (
                "Do not repeat the audio action. Inspect error.result.observation and verify "
                "the control is no longer held before continuing."
            ),
            result=result,
            followup_errors=followup_errors,
        )


@dataclass(frozen=True)
class WavInfo:
    """Validated PCM WAV metadata used to construct emulator packets."""

    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.sample_rate

    @property
    def sample_format(self) -> str:
        return "U8" if self.sample_width == 1 else "S16"


@dataclass(frozen=True)
class EmulatorEndpoint:
    """A local emulator control endpoint; its bearer token is always repr-hidden."""

    serial: str
    port: int
    token: str = field(repr=False)
    pid: int | None = None
    command_line: str = field(default="", repr=False)
    emulator_version: str = ""
    runtime_record: Path | None = field(default=None, repr=False)


@dataclass(frozen=True)
class PreparedInjection:
    """All failure-prone preflight state needed before a hold gesture begins."""

    wav: WavInfo
    endpoint: EmulatorEndpoint
    grpc: Any = field(repr=False)
    attempt_guard: Path | None = None
    attempt_claimed: bool = field(default=False, repr=False)


def _invalid_wav(_path: Path, detail: str) -> UsageError:
    return UsageError(
        f"cannot use the supplied WAV file: {detail}",
        code="mic_wav_invalid",
        hint="Provide an uncompressed PCM WAV: U8 or little-endian S16, mono or stereo, at 48 kHz or less.",
    )


def inspect_pcm_wav(path: str | Path) -> WavInfo:
    """Validate the emulator's intentionally narrow WAV input contract."""

    wav_path = Path(path).expanduser()
    if not wav_path.is_file():
        raise UsageError(
            "the supplied WAV file was not found",
            code="mic_wav_not_found",
            hint="Pass the path to a readable PCM WAV file.",
        )
    try:
        with wave.open(str(wav_path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            compression = source.getcomptype()
    except (EOFError, OSError, wave.Error) as exc:
        raise _invalid_wav(wav_path, "the file is malformed or is not a RIFF/WAVE file") from exc

    if compression != "NONE":
        raise _invalid_wav(wav_path, "compressed WAV audio is not supported")
    if channels not in {1, 2}:
        raise _invalid_wav(wav_path, f"expected mono or stereo audio, found {channels} channels")
    if sample_width not in {1, 2}:
        bits = sample_width * 8
        raise _invalid_wav(
            wav_path, f"expected unsigned 8-bit or signed 16-bit PCM, found {bits}-bit"
        )
    if not 1 <= sample_rate <= _MAX_SAMPLE_RATE:
        raise _invalid_wav(
            wav_path,
            f"sample rate must be between 1 and {_MAX_SAMPLE_RATE} Hz, found {sample_rate} Hz",
        )
    if frame_count <= 0:
        raise _invalid_wav(wav_path, "the audio stream is empty")
    duration_s = frame_count / sample_rate
    if duration_s > MAX_WAV_DURATION_S:
        raise _invalid_wav(
            wav_path,
            f"audio duration must be at most {MAX_WAV_DURATION_S:.0f} seconds, "
            f"found {duration_s:.3f} seconds",
        )

    return WavInfo(
        path=wav_path.resolve(),
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frame_count=frame_count,
    )


def _varint(value: int) -> bytes:
    if value < 0:  # pragma: no cover - all callers use unsigned metadata
        raise ValueError("protobuf varints must be non-negative")
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _uint_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _bytes_field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def audio_format_message(wav: WavInfo) -> bytes:
    """Encode ``android.emulation.control.AudioFormat`` without generated bindings."""

    message = _uint_field(1, wav.sample_rate)
    # Proto3 defaults encode Mono and U8 as absent zero-valued enum fields.
    if wav.channels == 2:
        message += _uint_field(2, 1)
    if wav.sample_width == 2:
        message += _uint_field(3, 1)
    # DeliveryMode remains MODE_UNSPECIFIED: the server then backpressures the stream.
    return message


def audio_packet_message(wav: WavInfo, audio: bytes) -> bytes:
    """Encode one ``AudioPacket`` with format and raw interleaved PCM bytes."""

    return _bytes_field(1, audio_format_message(wav)) + _bytes_field(3, audio)


def iter_audio_packets(wav: WavInfo, *, chunk_ms: int = _CHUNK_MS) -> Iterator[bytes]:
    """Yield small packets so every write fits comfortably in the emulator buffer."""

    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    frames_per_chunk = max(1, wav.sample_rate * chunk_ms // 1000)
    bytes_per_frame = wav.channels * wav.sample_width
    try:
        with wave.open(str(wav.path), "rb") as source:
            current_format = (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                source.getnframes(),
                source.getcomptype(),
            )
            expected_format = (
                wav.sample_rate,
                wav.channels,
                wav.sample_width,
                wav.frame_count,
                "NONE",
            )
            if current_format != expected_format:
                raise _invalid_wav(wav.path, "the audio metadata changed after validation")

            # Snapshot the bounded, validated payload before yielding the first packet.  That
            # prevents a path replacement or in-place edit from changing later chunks after
            # the server has already consumed part of the stream.
            audio = source.readframes(wav.frame_count)
            if len(audio) != wav.frame_count * bytes_per_frame:
                raise _invalid_wav(wav.path, "the audio data changed after validation")
    except (EOFError, OSError, wave.Error) as exc:
        # The file can change between validation and streaming. Keep that failure focused.
        raise _invalid_wav(wav.path, "the audio stream became unreadable") from exc

    bytes_per_chunk = frames_per_chunk * bytes_per_frame
    for offset in range(0, len(audio), bytes_per_chunk):
        yield audio_packet_message(wav, audio[offset : offset + bytes_per_chunk])


def _runtime_roots(
    *,
    system: str | None = None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> list[Path]:
    system = system or platform.system()
    env = os.environ if environ is None else environ
    user_home = Path.home() if home is None else home
    roots: list[Path] = []

    if system == "Darwin":
        roots.append(user_home / "Library/Caches/TemporaryItems")
    elif system == "Windows":
        if local_app_data := env.get("LOCALAPPDATA"):
            roots.append(Path(local_app_data) / "Temp")
    else:
        if runtime_dir := env.get("XDG_RUNTIME_DIR"):
            roots.append(Path(runtime_dir))
        with contextlib.suppress(AttributeError):  # pragma: no cover - non-POSIX fallback
            roots.append(Path(f"/run/user/{os.getuid()}"))
        roots.append(Path("/tmp"))

    for name in ("ANDROID_EMULATOR_HOME", "ANDROID_AVD_HOME"):
        if configured := env.get(name):
            roots.append(Path(configured))
    if sdk_home := env.get("ANDROID_SDK_HOME"):
        roots.append(Path(sdk_home) / ".android")
    roots.append(user_home / ".android")

    # Preserve preference while avoiding repeated scans of equivalent locations.
    return list(dict.fromkeys(root.expanduser() for root in roots))


def _candidate_running_dirs(roots: Sequence[Path]) -> Iterator[Path]:
    for root in roots:
        if root.name == "running" and root.parent.name == "avd":
            yield root
        else:
            yield root / "avd/running"


def _read_ini(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = raw_line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
    except OSError:
        return {}
    return values


def _configure_windows_kernel32(kernel32: Any) -> Any:
    """Give ctypes Win32 calls pointer-safe signatures on 64-bit hosts."""

    import ctypes
    from ctypes import wintypes

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _windows_pid_is_live(pid: int, *, kernel32: Any | None = None) -> bool:
    """Query a Windows process without sending a signal (which could terminate it)."""

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    if kernel32 is None:
        win_dll = ctypes.WinDLL  # type: ignore[attr-defined]
        kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32 = _configure_windows_kernel32(kernel32)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    exit_code = wintypes.DWORD()
    try:
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _pid_is_live(pid: int, *, system: str | None = None) -> bool:
    """Return whether a runtime record still names a host process."""

    if pid <= 0:
        return False
    if (system or platform.system()) == "Windows":
        # CPython's os.kill on Windows delegates most signals (including 0) to
        # TerminateProcess. A Unix-style liveness probe could kill the emulator.
        return _windows_pid_is_live(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _command_has_flag(command_line: str, flag: str) -> bool:
    """Match one emulator argv flag even though pid ini records quote every argument."""

    try:
        argv = shlex.split(command_line)
    except ValueError:
        argv = command_line.split()
    return any(value.strip("'\"") == flag for value in argv)


def discover_emulator_endpoint(
    serial: str,
    *,
    running_dirs: Sequence[Path] | None = None,
    pid_is_live: Callable[[int], bool] = _pid_is_live,
) -> EmulatorEndpoint:
    """Find the authenticated gRPC endpoint belonging to one adb emulator serial."""

    if not serial.startswith("emulator-"):
        raise UsageError(
            f"microphone injection requires an Android Emulator; selected device is '{serial}'",
            code="mic_requires_emulator",
            hint="Select an emulator serial such as emulator-5554. Physical devices do not expose the emulator control API.",
        )
    try:
        adb_port = int(serial.removeprefix("emulator-"))
    except ValueError as exc:
        raise UsageError(
            f"cannot identify emulator port from serial '{serial}'",
            code="mic_requires_emulator",
        ) from exc

    directories = (
        list(running_dirs)
        if running_dirs is not None
        else list(_candidate_running_dirs(_runtime_roots()))
    )
    matches: list[tuple[int, Path, dict[str, str]]] = []
    saw_serial = False
    for directory in directories:
        try:
            candidates = sorted(directory.glob("pid_*.ini"))
        except OSError:
            continue
        for candidate in candidates:
            values = _read_ini(candidate)
            if values.get("port.serial") != str(adb_port):
                continue
            saw_serial = True
            raw_pid = candidate.stem.removeprefix("pid_")
            if not raw_pid.isdigit():
                continue
            pid = int(raw_pid)
            if not pid_is_live(pid):
                continue
            matches.append((pid, candidate, values))

    if matches:
        # More than one live record should not normally claim one console serial. Prefer the
        # newest PID only after dead records have been discarded.
        pid, runtime_record, values = max(matches, key=lambda match: match[0])
        command_line = values.get("cmdline", "")
        if _command_has_flag(command_line, "-no-audio"):
            raise UsageError(
                f"emulator '{serial}' was started with -no-audio",
                code="mic_audio_disabled",
                hint="Restart that emulator with `aua emulator start --audio` (or start it without -no-audio), then retry.",
            )
        raw_port = values.get("grpc.port", "")
        token = values.get("grpc.token", "")
        try:
            grpc_port = int(raw_port)
        except ValueError:
            grpc_port = 0
        if 1 <= grpc_port <= 65_535 and token:
            return EmulatorEndpoint(
                serial=serial,
                port=grpc_port,
                token=token,
                pid=pid,
                command_line=command_line,
                emulator_version=values.get("emulator.version", ""),
                runtime_record=runtime_record.resolve(),
            )
    detail = "does not publish a usable authenticated gRPC endpoint"
    if not saw_serial:
        detail = "has no matching emulator discovery record"
    raise DeviceError(
        f"emulator '{serial}' {detail}",
        code="mic_endpoint_missing",
        hint="Use a recent Android Emulator, keep its authenticated gRPC endpoint enabled, and retry after it is fully started.",
    )


def _single_injection_only(version: str) -> bool:
    parts = version.strip().split(".")
    try:
        parsed = tuple(int(part) for part in parts[:3])
    except ValueError:
        return False
    return parsed == _AFFECTED_SINGLE_INJECTION_VERSION


def _attempt_guard_path(endpoint: EmulatorEndpoint) -> Path | None:
    if not _single_injection_only(endpoint.emulator_version) or endpoint.pid is None:
        return None
    if endpoint.runtime_record is None:
        raise DeviceError(
            "cannot identify this emulator boot for safe microphone injection",
            code="mic_guard_unavailable",
            hint=(
                "AUA fails closed on this emulator build. Restart only the emulator and make "
                "sure its pid_*.ini runtime record is readable."
            ),
        )
    safe_serial = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in endpoint.serial
    )
    # The endpoint token is random per emulator process. Hashing it gives workers using
    # different AUA cache directories the same boot identity without disclosing the token.
    boot_fingerprint = hashlib.sha256(endpoint.token.encode("utf-8")).hexdigest()[:24]
    return (
        endpoint.runtime_record.parent
        / f".aua-{safe_serial}-pid-{endpoint.pid}-{boot_fingerprint}.inject-audio-attempted"
    )


def claim_injection_attempt(prepared: PreparedInjection) -> PreparedInjection:
    """Atomically reserve the one safe stream attempt on affected emulator builds."""

    guard = prepared.attempt_guard
    if guard is None or prepared.attempt_claimed:
        return prepared
    try:
        guard.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        version = prepared.endpoint.emulator_version or "unknown"
        raise DeviceError(
            f"Android Emulator {version} cannot safely accept a second microphone injection "
            f"during the same emulator boot",
            code="mic_repeat_unsafe",
            hint=(
                "Do not retry. Restart only this emulator with `--audio`, reopen the target "
                "screen, and make the next injection the only attempt in that boot."
            ),
        ) from exc
    except OSError as exc:
        raise DeviceError(
            "could not persist the emulator microphone safety guard",
            code="mic_guard_unavailable",
            hint=(
                "AUA fails closed on this emulator build. Check emulator runtime-directory permissions, "
                "then restart only the emulator before trying once."
            ),
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                f"version={prepared.endpoint.emulator_version}\n"
                f"pid={prepared.endpoint.pid}\n"
                f"claimed_at={time.time():.6f}\n"
            )
    except OSError as exc:
        with contextlib.suppress(OSError):
            guard.unlink()
        raise DeviceError(
            "could not persist the emulator microphone safety guard",
            code="mic_guard_unavailable",
            hint=(
                "AUA fails closed on this emulator build. Check emulator runtime-directory permissions, "
                "then restart only the emulator before trying once."
            ),
        ) from exc
    return replace(prepared, attempt_claimed=True)


def _load_grpc() -> Any:
    try:
        import grpc
    except ImportError as exc:
        raise UsageError(
            "microphone injection needs the optional grpcio dependency",
            code="mic_grpc_unavailable",
            hint="Install it with `pip install 'android-ui-analyser[audio]'` (or the equivalent uv tool extra).",
        ) from exc
    return grpc


def prepare_injection(
    serial: str,
    path: str | Path,
    *,
    running_dirs: Sequence[Path] | None = None,
    grpc_module: Any | None = None,
) -> PreparedInjection:
    """Validate file, endpoint, and optional dependency before changing device input."""

    wav = inspect_pcm_wav(path)
    endpoint = discover_emulator_endpoint(serial, running_dirs=running_dirs)
    grpc = grpc_module if grpc_module is not None else _load_grpc()
    return PreparedInjection(
        wav=wav,
        endpoint=endpoint,
        grpc=grpc,
        attempt_guard=_attempt_guard_path(endpoint),
    )


def _status_name(exc: Exception) -> str:
    try:
        status = exc.code()  # type: ignore[attr-defined]
    except Exception:
        return "UNKNOWN"
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    return str(status).rsplit(".", 1)[-1]


def inject_prepared(prepared: PreparedInjection) -> WavInfo:
    """Run the authenticated stream and translate gRPC failures without leaking its token."""

    prepared = claim_injection_attempt(prepared)
    grpc = prepared.grpc
    endpoint = prepared.endpoint
    # The emulator endpoint is strictly loopback.  Disable gRPC's environment-proxy
    # discovery so its bearer token and PCM stream can never be offered to a configured
    # HTTP(S) proxy, even when localhost is missing from NO_PROXY.
    channel = grpc.insecure_channel(
        f"127.0.0.1:{endpoint.port}",
        options=(("grpc.enable_http_proxy", 0),),
    )
    timeout_s = max(10.0, prepared.wav.duration_s + 10.0)
    try:
        inject = channel.stream_unary(
            _INJECT_AUDIO_METHOD,
            request_serializer=lambda packet: packet,
            response_deserializer=lambda _response: None,
        )
        inject(
            iter_audio_packets(prepared.wav),
            metadata=(("authorization", f"Bearer {endpoint.token}"),),
            timeout=timeout_s,
        )
    except Exception as exc:
        rpc_error_type = getattr(grpc, "RpcError", ())
        if not isinstance(exc, rpc_error_type):
            raise
        status = _status_name(exc)
        if status == "FAILED_PRECONDITION":
            raise DeviceError(
                "the emulator could not activate audio injection because another microphone input is active or microphone registration failed",
                code="mic_injection_precondition",
                hint="Close other emulator microphone streams, make sure the app is requesting microphone input, and retry.",
            ) from exc
        if status == "INVALID_ARGUMENT":
            raise DeviceError(
                "the emulator rejected the audio stream format or packet size",
                code="mic_injection_rejected",
                hint="Use PCM U8/S16 mono/stereo audio at 48 kHz or less.",
            ) from exc
        if status in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
            raise DeviceError(
                "the emulator rejected authentication for its audio endpoint",
                code="mic_endpoint_auth_failed",
                hint="Restart the emulator so AUA can discover a fresh control token, then retry.",
            ) from exc
        if status == "DEADLINE_EXCEEDED":
            raise DeviceError(
                "audio injection timed out while waiting for the emulator to consume microphone samples",
                code="mic_injection_timeout",
                hint=(
                    "Samples may already have arrived. Do not retry blindly: inspect the "
                    "current UI first, then restart only the emulator if it is unresponsive."
                ),
            ) from exc
        if status == "UNAVAILABLE":
            raise DeviceError(
                "the emulator audio endpoint became unavailable; the emulator may have exited or gone offline",
                code="mic_emulator_unavailable",
                hint=(
                    "Do not retry the injection blindly. Run `aua devices`; if the emulator is "
                    "offline or absent, restart only that emulator with audio enabled."
                ),
            ) from exc
        if status == "INTERNAL":
            # Emulator 36.4.10 has returned INTERNAL only after the app received non-silent
            # samples and completed its voice flow. Retrying would duplicate user input and can
            # also trigger an upstream emulator audio crash, so this is its own uncertain state.
            raise MicDeliveryUncertainError() from exc
        raise DeviceError(
            f"emulator audio injection failed ({status})",
            code="mic_injection_failed",
            hint=(
                "Samples may already have arrived. Do not retry blindly: inspect the current "
                "UI first and check that the emulator is responsive."
            ),
        ) from exc
    finally:
        close = getattr(channel, "close", None)
        if callable(close):
            close()
    return prepared.wav


def inject_wav(
    serial: str,
    path: str | Path,
    *,
    running_dirs: Sequence[Path] | None = None,
    grpc_module: Any | None = None,
) -> WavInfo:
    """Convenience wrapper for callers that do not need gesture preflight."""

    return inject_prepared(
        prepare_injection(
            serial,
            path,
            running_dirs=running_dirs,
            grpc_module=grpc_module,
        )
    )


def synthesize_speech(
    text: str,
    destination: str | Path,
    *,
    voice: str | None = None,
    rate: int | None = None,
    system: str | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """Use macOS ``say`` to create a deterministic PCM WAV for :func:`inject_wav`."""

    if not text.strip():
        raise UsageError("speech text cannot be empty", code="mic_speech_empty")
    host = platform.system() if system is None else system
    if host != "Darwin" or not Path("/usr/bin/say").is_file():
        raise UsageError(
            "`aua mic speak` currently requires macOS and /usr/bin/say",
            code="mic_speech_unsupported_host",
            hint="Generate a PCM WAV with a host TTS tool, then use `aua mic inject FILE.wav`.",
        )
    if rate is not None and rate <= 0:
        raise UsageError("speech rate must be greater than zero", code="mic_speech_rate_invalid")

    output = Path(destination)
    command = [
        "/usr/bin/say",
        "-o",
        str(output),
        "--file-format=WAVE",
        "--data-format=LEI16@44100",
        "--channels=1",
    ]
    if voice:
        command.extend(["--voice", voice])
    if rate is not None:
        command.extend(["--rate", str(rate)])
    try:
        # With no positional text, `say` reads stdin. This keeps potentially sensitive speech
        # out of the host process list while retaining ordinary subprocess argument safety.
        completed = run(
            command,
            input=text,
            capture_output=True,
            text=True,
            timeout=SPEECH_SYNTHESIS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceError(
            "macOS speech synthesis failed",
            code="mic_speech_failed",
            hint="Check that /usr/bin/say can synthesize the requested voice, then retry.",
        ) from exc
    if completed.returncode != 0 or not output.is_file():
        raise DeviceError(
            "macOS speech synthesis failed",
            code="mic_speech_failed",
            hint="Check the voice name and rate, or use `aua mic inject` with an existing PCM WAV.",
        )
    # Validate what the platform produced before the caller changes device state.
    inspect_pcm_wav(output)
    return output
