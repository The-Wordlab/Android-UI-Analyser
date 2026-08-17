"""Guarded local FunctionGemma policy selector.

The provider accepts an existing absolute base-model directory plus either the
small adapter shipped with AUA or an existing absolute LoRA adapter directory.
It never passes a repository ID to ``mlx_lm.load``, so provider activation cannot
silently download weights.  MLX imports and model loading remain lazy, and every
generation or protocol failure returns ``None`` for fail-closed handling by the
policy core.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ...policy import PolicyContext, guard_candidates, policy_messages, policy_tools
from ..base import Availability, PolicyProvider
from ..registry import register_policy

_STRICT_CALL = re.compile(
    r"\s*<start_function_call>call:select_candidate\{candidate_id:(-?[0-9]+)\}"
    r"(?:<end_function_call>)?\s*"
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
DEFAULT_MAX_TOKENS = 24
BUNDLED_ADAPTER = "bundled"
BUNDLED_MANIFEST = "manifest.json"
PROMPT_SCHEMA_NAME = "functiongemma-aua-candidate-policy-v3"
PROMPT_CANDIDATE_COUNT = 4
PROMPT_CANDIDATE_IDS = "dense opaque integers 0 through candidate_count minus 1"
ROLLOUT_MODES = ("shadow", "advisory")


class SelectionProtocolError(ValueError):
    """Raised internally when model output is not one canonical offered-ID call."""


def _prompt_candidate_counts(prompt_schema: Mapping[str, Any]) -> tuple[int, ...]:
    """Return authenticated cardinalities while retaining the frozen v3 schema."""

    if (
        prompt_schema.get("name") != PROMPT_SCHEMA_NAME
        or prompt_schema.get("candidate_ids") != PROMPT_CANDIDATE_IDS
    ):
        raise ValueError("FunctionGemma manifest prompt_schema is incompatible")
    legacy = prompt_schema.get("candidate_count")
    authored = prompt_schema.get("candidate_counts")
    if legacy is not None and authored is not None:
        raise ValueError("FunctionGemma prompt_schema cannot declare two cardinality formats")
    values: Any = (legacy,) if legacy is not None else authored
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("FunctionGemma prompt_schema lacks candidate cardinalities")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value not in range(2, 5)
        for value in values
    ):
        raise ValueError("FunctionGemma candidate cardinalities must be integers from 2 to 4")
    counts = tuple(sorted(set(values)))
    if len(counts) != len(values):
        raise ValueError("FunctionGemma candidate cardinalities must be unique")
    return counts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Hash relative file names and contents into one stable model identity."""

    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError("model directory contains no files")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(_file_sha256(candidate).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_model_sha256(file_hashes: Mapping[str, str]) -> str:
    """Hash a required runtime file set independently of unrelated snapshot extras."""

    digest = hashlib.sha256()
    for relative, file_hash in sorted(file_hashes.items()):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_manifest_model_files(
    model: Path,
    files: Any,
) -> tuple[str, tuple[str, ...], int]:
    """Verify every declared runtime file while deliberately ignoring extra files."""

    if not isinstance(files, Mapping) or not files:
        raise ValueError("bundled manifest base_model.files must be a non-empty mapping")
    actual_hashes: dict[str, str] = {}
    total_bytes = 0
    for raw_relative, raw_identity in files.items():
        if not isinstance(raw_relative, str) or not raw_relative:
            raise ValueError("bundled manifest model file paths must be non-empty strings")
        relative = PurePosixPath(raw_relative)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw_relative:
            raise ValueError("bundled manifest model file path is not canonical and relative")
        if not isinstance(raw_identity, Mapping):
            raise ValueError(f"bundled manifest identity for {raw_relative!r} must be an object")
        expected_hash = _manifest_sha256(
            raw_identity.get("sha256"), f"base_model.files.{raw_relative}.sha256"
        )
        expected_bytes = raw_identity.get("bytes")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            raise ValueError(
                f"bundled manifest base_model.files.{raw_relative}.bytes must be non-negative"
            )
        candidate = model.joinpath(*relative.parts)
        if not candidate.is_file():
            raise ValueError(f"required base-model runtime file is missing: {raw_relative}")
        actual_bytes = candidate.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"base-model runtime file byte size mismatch: {raw_relative}")
        actual_hash = _file_sha256(candidate)
        if actual_hash != expected_hash:
            raise ValueError(f"base-model runtime file SHA-256 mismatch: {raw_relative}")
        actual_hashes[raw_relative] = actual_hash
        total_bytes += actual_bytes
    return (
        _canonical_model_sha256(actual_hashes),
        tuple(sorted(actual_hashes)),
        total_bytes,
    )


def _absolute_directory(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} is not configured")
    authored = Path(value)
    if not authored.is_absolute():
        raise ValueError(f"{label} must be an absolute local path")
    path = authored.resolve()
    if not path.is_dir():
        raise ValueError(f"{label} is not an existing directory")
    return path


def bundled_adapter_path() -> Path:
    """Return the installed package directory containing the small shipped adapter."""

    package_root = Path(__file__).resolve().parents[2]
    return package_root / "resources" / "functiongemma"


def _uses_bundled_adapter(settings: Mapping[str, Any]) -> bool:
    value = settings.get("adapter_path")
    return value is None or value == "" or value == BUNDLED_ADAPTER


def _manifest_file(directory: Path, value: Any, default: str) -> Path:
    relative = Path(str(value or default))
    if relative.is_absolute():
        raise ValueError("bundled adapter manifest paths must be relative")
    resolved = (directory / relative).resolve()
    try:
        resolved.relative_to(directory)
    except ValueError as exc:
        raise ValueError("bundled adapter manifest path escapes its resource directory") from exc
    return resolved


def _bundled_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / BUNDLED_MANIFEST
    if not manifest_path.is_file():
        raise ValueError("bundled FunctionGemma adapter manifest is missing")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("bundled FunctionGemma adapter manifest is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported bundled FunctionGemma adapter manifest")
    if not isinstance(value.get("base_model"), Mapping) or not isinstance(
        value.get("adapter"), Mapping
    ):
        raise ValueError("bundled FunctionGemma manifest lacks base_model or adapter provenance")
    prompt_schema = value.get("prompt_schema")
    if not isinstance(prompt_schema, Mapping):
        raise ValueError("bundled FunctionGemma manifest prompt_schema is incompatible")
    try:
        _prompt_candidate_counts(prompt_schema)
    except ValueError as exc:
        raise ValueError("bundled FunctionGemma manifest prompt_schema is incompatible") from exc
    rollout = value.get("rollout")
    if not isinstance(rollout, Mapping) or rollout.get("max_mode") not in ROLLOUT_MODES:
        raise ValueError("bundled FunctionGemma manifest rollout.max_mode is invalid")
    return value


def _explicit_rollout_capability(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Read a pinned explicit-adapter rollout manifest without hashing model weights."""

    base = {
        "authenticated": False,
        "max_mode": "shadow",
        "source": "explicit_adapter",
        "reason": (
            "explicit adapters are shadow-only without a manifest pinned by manifest_sha256, "
            "model_sha256, and adapter_sha256"
        ),
    }
    try:
        adapter = _absolute_directory(settings.get("adapter_path"), "adapter_path")
        expected_manifest_hash = _expected_sha256(settings, "manifest_sha256")
        expected_model_hash = _expected_sha256(settings, "model_sha256")
        expected_adapter_hash = _expected_sha256(settings, "adapter_sha256")
        if not expected_manifest_hash or not expected_model_hash or not expected_adapter_hash:
            return base
        manifest_path = adapter / BUNDLED_MANIFEST
        if not manifest_path.is_file():
            return {**base, "reason": "pinned explicit rollout manifest is missing"}
        actual_manifest_hash = _file_sha256(manifest_path)
        if actual_manifest_hash != expected_manifest_hash:
            return {**base, "reason": "explicit rollout manifest SHA-256 does not match pin"}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {**base, "reason": "explicit rollout manifest is not valid JSON"}
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
            return {**base, "reason": "explicit rollout manifest schema is unsupported"}
        rollout = manifest.get("rollout")
        prompt_schema = manifest.get("prompt_schema")
        base_model = manifest.get("base_model")
        adapter_value = manifest.get("adapter")
        if (
            not isinstance(rollout, Mapping)
            or rollout.get("max_mode") not in ROLLOUT_MODES
            or not isinstance(prompt_schema, Mapping)
            or not isinstance(base_model, Mapping)
            or not isinstance(adapter_value, Mapping)
        ):
            return {**base, "reason": "explicit rollout manifest capability is incomplete"}
        try:
            _prompt_candidate_counts(prompt_schema)
        except ValueError:
            return {**base, "reason": "explicit rollout manifest capability is incomplete"}
        manifest_model_hash = _manifest_sha256(base_model.get("sha256"), "base_model.sha256")
        manifest_adapter_hash = _manifest_sha256(adapter_value.get("sha256"), "adapter.sha256")
        if (
            manifest_model_hash != expected_model_hash
            or manifest_adapter_hash != expected_adapter_hash
        ):
            return {
                **base,
                "reason": "explicit rollout manifest is not bound to configured artifacts",
            }
        return {
            "authenticated": True,
            "max_mode": str(rollout["max_mode"]),
            "source": "pinned_explicit_manifest",
            "reason": "explicit rollout manifest and artifact identities are pinned",
            "manifest_sha256": actual_manifest_hash,
            "manifest": manifest,
        }
    except Exception as exc:
        return {**base, "reason": f"explicit rollout capability is invalid: {exc}"}


def _manifest_sha256(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"bundled manifest {label} must be a SHA-256 digest")
    return text


def _bundled_artifact_signature(settings: Mapping[str, Any]) -> tuple[tuple[str, int, int], ...]:
    """Cheap cache key over every manifest-required input file."""

    model = _absolute_directory(settings.get("model_path"), "model_path")
    adapter = _absolute_directory(bundled_adapter_path(), "bundled adapter_path")
    manifest = _bundled_manifest(adapter)
    manifest_model = manifest["base_model"]
    files = manifest_model.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("bundled manifest base_model.files must be a non-empty mapping")
    paths = [adapter / BUNDLED_MANIFEST]
    manifest_adapter = manifest["adapter"]
    paths.extend(
        (
            _manifest_file(adapter, manifest_adapter.get("config"), "adapter_config.json"),
            _manifest_file(adapter, manifest_adapter.get("weights"), "adapters.safetensors"),
        )
    )
    for raw_relative in files:
        if not isinstance(raw_relative, str) or not raw_relative:
            raise ValueError("bundled manifest model file paths must be non-empty strings")
        relative = PurePosixPath(raw_relative)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw_relative:
            raise ValueError("bundled manifest model file path is not canonical and relative")
        paths.append(model.joinpath(*relative.parts))
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"required policy artifact is missing: {path.name}")
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    settings_identity = json.dumps(
        {
            "model_path": str(settings.get("model_path") or ""),
            "model_sha256": settings.get("model_sha256"),
            "adapter_sha256": settings.get("adapter_sha256"),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    signature.append((f"settings:{hashlib.sha256(settings_identity.encode()).hexdigest()}", 0, 0))
    return tuple(sorted(signature))


def _expected_sha256(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings.get(key)
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{key} must contain exactly 64 hexadecimal characters")
    return text


def validate_local_artifacts(
    settings: Mapping[str, Any],
    *,
    include_model_hash: bool = False,
) -> dict[str, Any]:
    """Validate local-only base/LoRA provenance without importing or loading MLX."""

    model = _absolute_directory(settings.get("model_path"), "model_path")
    adapter_setting = settings.get("adapter_path")
    bundled = _uses_bundled_adapter(settings)
    adapter = (
        _absolute_directory(bundled_adapter_path(), "bundled adapter_path")
        if bundled
        else _absolute_directory(adapter_setting, "adapter_path")
    )
    manifest = _bundled_manifest(adapter) if bundled else None
    if not bundled and settings.get("manifest_sha256") not in {None, ""}:
        explicit_capability = _explicit_rollout_capability(settings)
        if not explicit_capability.get("authenticated"):
            raise ValueError(str(explicit_capability.get("reason") or "invalid rollout capability"))
        explicit_manifest = explicit_capability.get("manifest")
        if not isinstance(explicit_manifest, Mapping):
            raise ValueError("authenticated explicit rollout manifest is unavailable")
        manifest = dict(explicit_manifest)
    model_config_path = model / "config.json"
    if not model_config_path.is_file():
        raise ValueError("model_path does not contain config.json")

    manifest_adapter = manifest["adapter"] if manifest is not None else {}
    adapter_config_path = _manifest_file(
        adapter, manifest_adapter.get("config"), "adapter_config.json"
    )
    weights_path = _manifest_file(adapter, manifest_adapter.get("weights"), "adapters.safetensors")
    if not adapter_config_path.is_file() or not weights_path.is_file():
        raise ValueError("adapter_path is missing adapter_config.json or adapters.safetensors")
    if weights_path.stat().st_size <= 0:
        raise ValueError("adapter weights are empty")
    try:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("adapter_config.json is not valid JSON") from exc
    if not isinstance(adapter_config, Mapping):
        raise ValueError("adapter_config.json must contain an object")
    if str(adapter_config.get("fine_tune_type") or "").strip().lower() != "lora":
        raise ValueError("adapter_config.json must identify a LoRA adapter")

    actual_adapter_config_hash = _file_sha256(adapter_config_path)
    if manifest is not None:
        expected_config_hash = _manifest_sha256(
            manifest_adapter.get("config_sha256"), "adapter.config_sha256"
        )
        if actual_adapter_config_hash != expected_config_hash:
            raise ValueError("adapter config SHA-256 does not match rollout manifest")
        manifest_fine_tune_type = manifest_adapter.get("fine_tune_type")
        if manifest_fine_tune_type is not None and (
            str(manifest_fine_tune_type).strip().lower()
            != str(adapter_config.get("fine_tune_type")).strip().lower()
        ):
            raise ValueError("rollout manifest fine_tune_type does not match adapter config")
        lora_parameters = adapter_config.get("lora_parameters")
        for field in ("rank", "scale", "dropout"):
            if field not in manifest_adapter:
                continue
            if not isinstance(lora_parameters, Mapping) or (
                manifest_adapter[field] != lora_parameters.get(field)
            ):
                raise ValueError(
                    f"rollout manifest {field} does not match adapter config lora_parameters"
                )

    if not bundled:
        configured_model_raw = adapter_config.get("model")
        if not isinstance(configured_model_raw, str) or not configured_model_raw.strip():
            raise ValueError("adapter_config.json does not identify its base model")
        configured_model = Path(configured_model_raw)
        if not configured_model.is_absolute() or not configured_model.is_dir():
            raise ValueError("adapter base model must be an existing absolute local path")
        if configured_model.resolve() != model:
            raise ValueError("adapter base model does not match configured model_path")

    expected_adapter_hash = _expected_sha256(settings, "adapter_sha256")
    manifest_adapter_hash = (
        _manifest_sha256(manifest_adapter.get("sha256"), "adapter.sha256")
        if manifest is not None
        else None
    )
    if (
        expected_adapter_hash
        and manifest_adapter_hash
        and expected_adapter_hash != manifest_adapter_hash
    ):
        raise ValueError("configured adapter_sha256 conflicts with rollout manifest")
    actual_adapter_hash = _file_sha256(weights_path)
    if (expected_adapter_hash or manifest_adapter_hash) not in {None, actual_adapter_hash}:
        raise ValueError("adapter_sha256 does not match adapters.safetensors")
    manifest_bytes = manifest_adapter.get("bytes") if manifest is not None else None
    if manifest_bytes is not None and (
        not isinstance(manifest_bytes, int)
        or isinstance(manifest_bytes, bool)
        or weights_path.stat().st_size != manifest_bytes
    ):
        raise ValueError("adapter byte size does not match rollout manifest")

    expected_model_hash = _expected_sha256(settings, "model_sha256")
    manifest_model = manifest["base_model"] if manifest is not None else {}
    manifest_model_hash = (
        _manifest_sha256(manifest_model.get("sha256"), "base_model.sha256")
        if manifest is not None
        else None
    )
    if expected_model_hash and manifest_model_hash and expected_model_hash != manifest_model_hash:
        raise ValueError("configured model_sha256 conflicts with rollout manifest")
    actual_model_hash: str | None = None
    required_model_files: tuple[str, ...] = ()
    required_model_bytes: int | None = None
    if bundled and manifest is not None:
        (
            actual_model_hash,
            required_model_files,
            required_model_bytes,
        ) = _validate_manifest_model_files(model, manifest_model.get("files"))
    elif expected_model_hash or include_model_hash:
        actual_model_hash = _tree_sha256(model)
    if (expected_model_hash or manifest_model_hash) not in {None, actual_model_hash}:
        raise ValueError("model_sha256 does not match model directory")

    return {
        "model_path": str(model),
        "adapter_path": str(adapter),
        "model_sha256": actual_model_hash,
        "model_hash_verified": bool(expected_model_hash or manifest_model_hash),
        "model_hash_kind": "required_runtime_files" if bundled else "directory_tree",
        "model_required_files": list(required_model_files),
        "model_required_bytes": required_model_bytes,
        "adapter_sha256": actual_adapter_hash,
        "adapter_hash_verified": bool(expected_adapter_hash or manifest_adapter_hash),
        "adapter_config_sha256": actual_adapter_config_hash,
        "adapter_bytes": weights_path.stat().st_size,
        "fine_tune_type": "lora",
        "adapter_source": "bundled" if bundled else "local",
        "manifest_path": str(adapter / BUNDLED_MANIFEST) if manifest is not None else None,
        "rollout_max_mode": (
            str(manifest["rollout"]["max_mode"])
            if manifest is not None and isinstance(manifest.get("rollout"), Mapping)
            else "shadow"
        ),
        "rollout_authenticated": bool(bundled or settings.get("manifest_sha256")),
    }


def parse_candidate_id(output: Any, tokenizer: Any, tools: list[dict[str, Any]]) -> int:
    """Parse exactly one canonical FunctionGemma call and reject all other output."""

    if not isinstance(output, str):
        raise SelectionProtocolError("model output is not text")
    strict_match = _STRICT_CALL.fullmatch(output)
    if strict_match is None:
        raise SelectionProtocolError("output is not exactly one canonical selector call")
    try:
        parsed = tokenizer.tool_parser(output, tools)
    except Exception as exc:
        raise SelectionProtocolError("FunctionGemma tool parser rejected output") from exc
    if not isinstance(parsed, Mapping) or parsed.get("name") != "select_candidate":
        raise SelectionProtocolError("output invoked an unexpected function")
    arguments = parsed.get("arguments")
    candidate_id = arguments.get("candidate_id") if isinstance(arguments, Mapping) else None
    if not isinstance(candidate_id, int) or isinstance(candidate_id, bool):
        raise SelectionProtocolError("candidate_id is not an integer")
    if candidate_id != int(strict_match.group(1)):
        raise SelectionProtocolError("protocol parser and strict call disagree")
    return candidate_id


def _validate_tokenizer(tokenizer: Any) -> None:
    if not bool(getattr(tokenizer, "has_chat_template", False)):
        raise ValueError("model tokenizer lacks a chat template")
    if not callable(getattr(tokenizer, "tool_parser", None)):
        raise ValueError("model tokenizer lacks FunctionGemma tool parsing")
    if not getattr(tokenizer, "tool_call_start", None) or not getattr(
        tokenizer, "tool_call_end", None
    ):
        raise ValueError("model tokenizer lacks FunctionGemma tool boundaries")
    smoke = "<start_function_call>call:select_candidate{candidate_id:3}"
    parsed = tokenizer.tool_parser(smoke, [])
    if parsed != {"name": "select_candidate", "arguments": {"candidate_id": 3}}:
        raise ValueError("FunctionGemma tool parser smoke failed")


@register_policy("functiongemma")
class FunctionGemmaPolicySelector(PolicyProvider):
    """Lazy, greedy MLX selector over AUA-authored opaque candidates."""

    def __init__(
        self,
        settings: Mapping[str, Any] | None = None,
        *,
        model_loader: Callable[..., Any] | None = None,
        generator: Callable[..., str] | None = None,
        sampler_factory: Callable[..., Any] | None = None,
        runtime_availability: Callable[[], Availability] | None = None,
    ) -> None:
        super().__init__(settings)
        self._model_loader = model_loader
        self._generator = generator
        self._sampler_factory = sampler_factory
        self._runtime_availability_override = runtime_availability
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._sampler: Any | None = None
        self._provenance: dict[str, Any] | None = None
        self._artifact_signature: tuple[tuple[str, int, int], ...] | None = None
        self.last_error: str | None = None

    def _max_tokens(self) -> int:
        value = self.settings.get("max_tokens", DEFAULT_MAX_TOKENS)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 64:
            raise ValueError("max_tokens must be an integer from 1 to 64")
        return value

    def _supported_candidate_counts(self) -> tuple[int, ...]:
        """Return counts authenticated by the selected adapter manifest."""

        if _uses_bundled_adapter(self.settings):
            manifest = _bundled_manifest(
                _absolute_directory(bundled_adapter_path(), "bundled adapter_path")
            )
            return _prompt_candidate_counts(manifest["prompt_schema"])
        capability = _explicit_rollout_capability(self.settings)
        manifest = capability.get("manifest")
        if capability.get("authenticated") and isinstance(manifest, Mapping):
            prompt_schema = manifest.get("prompt_schema")
            if isinstance(prompt_schema, Mapping):
                return _prompt_candidate_counts(prompt_schema)
        # Unauthenticated historical adapters remain compatible only with the
        # frozen exact-four shadow surface. Advisory is separately rejected.
        return (PROMPT_CANDIDATE_COUNT,)

    def supports_candidate_count(self, count: int) -> bool:
        """Return whether the authenticated adapter learned this cardinality."""

        try:
            return count in self._supported_candidate_counts()
        except Exception:
            return False

    def rollout_capability(self) -> dict[str, Any]:
        """Return the provenance-bound maximum mode without loading model artifacts."""

        if _uses_bundled_adapter(self.settings):
            try:
                manifest = _bundled_manifest(
                    _absolute_directory(bundled_adapter_path(), "bundled adapter_path")
                )
                max_mode = str(manifest["rollout"]["max_mode"])
                return {
                    "authenticated": True,
                    "max_mode": max_mode,
                    "supported_modes": list(ROLLOUT_MODES[: ROLLOUT_MODES.index(max_mode) + 1]),
                    "source": "bundled_manifest",
                    "reason": f"bundled manifest limits rollout to {max_mode}",
                }
            except Exception as exc:
                return {
                    "authenticated": False,
                    "max_mode": "shadow",
                    "supported_modes": ["shadow"],
                    "source": "bundled_manifest",
                    "reason": f"bundled rollout capability is invalid: {exc}",
                }
        capability = _explicit_rollout_capability(self.settings)
        capability.pop("manifest", None)
        max_mode = str(capability.get("max_mode") or "shadow")
        capability["supported_modes"] = list(ROLLOUT_MODES[: ROLLOUT_MODES.index(max_mode) + 1])
        return capability

    def supports_mode(self, mode: str) -> bool:
        """Advisory requires an authenticated capability explicitly allowing it."""

        if mode == "shadow":
            return True
        if mode != "advisory":
            return False
        capability = self.rollout_capability()
        return bool(capability.get("authenticated")) and capability.get("max_mode") == "advisory"

    def _runtime_availability(self) -> Availability:
        if self._runtime_availability_override is not None:
            return self._runtime_availability_override()
        if (
            self._model_loader is not None
            and self._generator is not None
            and self._sampler_factory is not None
        ):
            return Availability(True, "injected runtime")
        if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            return Availability(False, "FunctionGemma policy requires Apple silicon")
        if importlib.util.find_spec("mlx_lm") is None:
            return Availability(
                False,
                "optional dependency missing; install android-ui-analyser[functiongemma]",
            )
        return Availability(True, "local MLX runtime available")

    def _validated_artifacts(
        self,
        *,
        force: bool = False,
        include_model_hash: bool = False,
    ) -> dict[str, Any]:
        """Validate artifacts, caching only immutable-looking bundled snapshots.

        The cache key is deliberately cheap (resolved required paths, sizes, and
        nanosecond mtimes). A full hash verification is always forced immediately
        before first model load.
        """

        if not _uses_bundled_adapter(self.settings):
            provenance = validate_local_artifacts(
                self.settings, include_model_hash=include_model_hash
            )
            self._provenance = provenance
            self._artifact_signature = None
            return provenance

        before = _bundled_artifact_signature(self.settings)
        if not force and self._provenance is not None and before == self._artifact_signature:
            return self._provenance
        provenance = validate_local_artifacts(self.settings, include_model_hash=include_model_hash)
        after = _bundled_artifact_signature(self.settings)
        if before != after:
            raise ValueError("policy artifacts changed during provenance verification")
        self._provenance = provenance
        self._artifact_signature = after
        return provenance

    def is_available(self) -> Availability:
        """Validate platform, optional dependency, and local artifact provenance."""

        try:
            runtime = self._runtime_availability()
            if not runtime.ok:
                return runtime
            self._validated_artifacts()
            self._max_tokens()
        except Exception as exc:
            self._provenance = None
            self._artifact_signature = None
            self.last_error = str(exc)
            return Availability(False, str(exc))
        self.last_error = None
        return Availability(True, "local FunctionGemma model and LoRA adapter are ready")

    def provenance(self, *, include_model_hash: bool = False) -> dict[str, Any]:
        """Return local artifact identities without loading the model."""

        return self._validated_artifacts(force=True, include_model_hash=include_model_hash)

    def status(self) -> dict[str, Any]:
        """Return host-only diagnostics suitable for a CLI/MCP policy status surface."""

        rollout = self.rollout_capability()
        try:
            runtime = self._runtime_availability()
        except Exception:
            runtime = Availability(False, "policy runtime availability check failed")
        artifact_reason = "local artifacts are ready"
        artifacts_ready = False
        try:
            self._validated_artifacts()
            self._max_tokens()
            artifacts_ready = True
        except Exception as exc:
            artifact_reason = str(exc)
            self._provenance = None
            self._artifact_signature = None
        available = runtime.ok and artifacts_ready
        if available:
            reason = "local FunctionGemma model and LoRA adapter are ready"
        elif not artifacts_ready:
            reason = artifact_reason
        else:
            reason = runtime.reason
        self.last_error = None if available else reason
        return {
            "provider": self.name,
            "available": available,
            "reason": reason,
            "loaded": self._model is not None,
            "supported_candidate_counts": list(self._supported_candidate_counts()),
            "rollout": rollout,
            "last_error": self.last_error,
            "runtime": {"ready": runtime.ok, "reason": runtime.reason},
            "artifacts": {
                "ready": artifacts_ready,
                "reason": artifact_reason,
                "model_path": self.settings.get("model_path"),
                "adapter_path": self.settings.get("adapter_path"),
            },
            "provenance": dict(self._provenance or {}),
        }

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        # Revalidate immediately before first load to close the status-to-load gap.
        self._provenance = self._validated_artifacts(force=True)
        if self._model_loader is None or self._generator is None or self._sampler_factory is None:
            from mlx_lm import load
            from mlx_lm.generate import generate
            from mlx_lm.sample_utils import make_sampler

            self._model_loader = self._model_loader or load
            self._generator = self._generator or generate
            self._sampler_factory = self._sampler_factory or make_sampler

        model_path = self._provenance["model_path"]
        adapter_path = self._provenance["adapter_path"]
        loader = self._model_loader
        if loader is None:  # Defensive: all normal paths initialize it above.
            raise RuntimeError("FunctionGemma model loader is unavailable")
        loaded = loader(
            model_path,
            adapter_path=adapter_path,
        )
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            raise RuntimeError("FunctionGemma model loader returned an invalid result")
        self._model, self._tokenizer = loaded[:2]
        _validate_tokenizer(self._tokenizer)
        tool_call_end = getattr(self._tokenizer, "tool_call_end", None)
        if tool_call_end:
            self._tokenizer.add_eos_token(tool_call_end)
        assert self._sampler_factory is not None
        self._sampler = self._sampler_factory(temp=0.0)
        return self._model, self._tokenizer

    def select(self, context: PolicyContext) -> int | None:
        """Select one offered ID without exposing, rewriting, or executing its call."""

        # Defend the provider boundary even when a caller bypasses evaluate_policy.
        try:
            guarded = guard_candidates(
                context, max_candidates=max(1, min(4, len(context.candidates)))
            )
            max_tokens = self._max_tokens()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        if guarded != context.candidates:
            self.last_error = "provider received candidates that did not pass the policy guard"
            return None
        guarded_ids = {candidate.candidate_id for candidate in guarded}
        expected_ids = set(range(len(guarded)))
        if not self.supports_candidate_count(len(guarded)) or guarded_ids != expected_ids:
            self.last_error = (
                "the FunctionGemma adapter does not support this dense candidate cardinality"
            )
            return None

        try:
            model, tokenizer = self._ensure_loaded()
            tools = policy_tools()
            prompt = tokenizer.apply_chat_template(
                policy_messages(context, guarded),
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            assert self._generator is not None
            output = self._generator(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=self._sampler,
                verbose=False,
            )
            selected_id = parse_candidate_id(output, tokenizer, tools)
            offered_ids = {candidate.candidate_id for candidate in guarded}
            if selected_id not in offered_ids:
                raise SelectionProtocolError("model selected an ID that was not offered")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        self.last_error = None
        return selected_id
