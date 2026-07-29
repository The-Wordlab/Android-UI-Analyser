"""Configuration system (PRD §9): pydantic models, layered loading, profiles, secrets.

Precedence (highest first):
    1. individual CLI flags
    2. ``--config <path>`` file (replaces auto-discovered user/project files)
    3. environment variables (``AUA_*``; provider key vars are read at runtime, never here)
    4. ``--profile`` overlay
    5. project config: nearest ``.android-ui-analyser.yaml`` walking up from CWD
    6. user config: ``$XDG_CONFIG_HOME/android-ui-analyser/config.yaml``
    7. built-in defaults

Secrets are **never** stored in config: a provider references the env-var *name*
(``api_key_env: OPENAI_API_KEY``) and the value is read at runtime. ``config show`` and
``doctor`` never print secret values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError
from .schema import OutputFormat, Tier

PROJECT_CONFIG_NAME = ".android-ui-analyser.yaml"
USER_CONFIG_REL = "android-ui-analyser/config.yaml"
_SECRET_KEYS = {"api_key", "key", "token", "secret", "password"}


# --------------------------------------------------------------------------- models


class DeviceCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    serial: str | None = None  # null = auto-detect
    backend: str = "uiautomator2"  # uiautomator2 | accessibility (future)
    # Drop non-visible nodes in dump_hierarchy — smaller XML on deep trees.
    compressed_hierarchy: bool = True


class PerfCfg(BaseModel):
    """Latency knobs — defaults favour the agent hot path (tap → observe → tap)."""

    model_config = ConfigDict(extra="forbid")
    prefetch: bool = True  # background hierarchy dump after actions
    predictive_prefetch: bool = True  # kick prefetch during settle / from map edges
    async_memory: bool = True  # record screens/edges off the analyze critical path
    skip_unchanged_memory: bool = True  # skip map write when tree fingerprint unchanged
    reuse_capture_frames: bool = True  # share capture JPEGs with --with-image / settle
    capture_adb_screencap: bool = True  # capture loop prefers adb exec-out screencap
    differential: bool = False  # meta.element_diff vs previous analyze (token-cheap)
    auto_daemon: bool = True  # CLI auto-starts the warm daemon when enabled but down
    settle_profiles: bool = True  # learn per-action settle budgets from history
    gate_cache: bool = True  # memoize gate.decide for identical tree fingerprints


class GateCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_elements: int = 3
    min_labeled_ratio: float = 0.15
    vision_packages: list[str] = Field(
        default_factory=lambda: ["io.flutter", "com.unity3d", "org.libsdl", "*.WebView"]
    )
    # Soft gate for element-CLASS matches (e.g. `*.WebView` matching a WebView node): a
    # class match escalates to vision only when the tree is ALSO weak by these stricter
    # thresholds. Package/activity matches remain hard triggers.
    soft_min_elements: int = 8
    soft_min_labeled_ratio: float = 0.3


class WebviewCfg(BaseModel):
    """Try WebView DOM/a11y enrichment before escalating hollow WebViews to OCR."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    cdp: bool = False  # probe Chrome DevTools on :9222 when hierarchy children are empty
    min_elements: int = 3  # enrichment "good enough" to skip vision


class PerceptionCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gate: GateCfg = Field(default_factory=GateCfg)
    webview: WebviewCfg = Field(default_factory=WebviewCfg)


class RoutingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    auto_escalate: bool = True
    max_tier: Tier = Tier.vision
    semantic_query_hierarchy_first: bool = True

    @field_validator("max_tier", mode="before")
    @classmethod
    def _coerce_tier(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class OutputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: OutputFormat = OutputFormat.json
    annotate: bool = False
    # Session default for analyze/actions — per-call --with-image overrides when set.
    with_image: bool | str = False


class _ChainCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    chain: list[str] = Field(default_factory=list)


class OcrCfg(_ChainCfg):
    enabled: bool = True
    chain: list[str] = Field(default_factory=lambda: ["apple_vision", "rapidocr"])


class DetectionCfg(_ChainCfg):
    enabled: bool = True
    chain: list[str] = Field(default_factory=lambda: ["yolo", "omniparser"])


class GroundingCfg(_ChainCfg):
    enabled: bool = False  # opt-in (PRD §7.2)
    chain: list[str] = Field(default_factory=lambda: ["local_vllm", "gemini"])


class PlannerCfg(_ChainCfg):
    """Optional LLM navigator for goto/flow recovery + `aua navigate` (PRD §7.3)."""

    enabled: bool = False  # opt-in; also needs a per-call --assist (or `aua navigate`)
    chain: list[str] = Field(default_factory=lambda: ["gemini_flash"])


class TimeoutsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vision_ms: int = 8000  # OCR chain (fast)
    detection_ms: int = 20000  # detection chain (cold model load on per-call CLI can be slow)
    grounding_ms: int = 30000
    planner_ms: int = 15000  # per planner decision (fast text model)
    action_ms: int = 5000


class DaemonCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    socket: str = "~/.cache/android-ui-analyser/daemon.sock"


class CacheCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    dir: str = "~/.cache/android-ui-analyser"


class CaptureCfg(BaseModel):
    """Always-on rolling screencap buffer (daemon-warm sessions)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True  # auto-start with daemon
    idle_fps: float = 2.0
    burst_fps: float = 10.0  # best-effort; screencap often caps lower
    burst_ms: int = 1500
    extend_burst_on_change: bool = True  # keep bursting while pixels keep changing
    ttl_s: int = 180
    max_mb: int = 200
    jpeg_quality: int = 70
    hint: bool = True  # push meta.capture_hint after post-action pixel change
    sidecar: bool = True  # allow host capture sidecar when daemon is off


class MemoryCfg(BaseModel):
    """Persistent per-app map settings (PRD §6b, §9)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    auto_record: bool = True  # record screens + route edges on every analyze/action
    dir: str = "~/.android-ui-analyser"
    backend: Literal["json", "sqlite"] = "json"  # storage for AppMap + SessionState
    sqlite_path: str = "~/.android-ui-analyser/memory.db"  # used when backend=sqlite
    drift_threshold: float = 0.3  # signature divergence that flags a screen stale
    redact: bool = True  # never store secrets / PII / EditText values verbatim
    suggest: bool = True  # push known_routes/suggested_gotos/map_hint inline into analyze
    suggest_max: int = 4  # cap on suggested_gotos returned per analyze
    rank_half_life_days: float = 3.0  # recency decay for usage-based ranking (days)
    auto_research: bool = True  # materialize audit questions when map quality is uncertain
    research_suggest_max: int = 3  # cap research prompts pushed inline into analyze
    # Packages that never count as the foreground app: excluded from the hierarchy
    # package vote (an open keyboard must not win it) and never recorded as maps.
    ignore_packages: list[str] = Field(
        default_factory=lambda: ["com.android.systemui", "*inputmethod*"]
    )
    # Auth/consent surfaces a flow passes THROUGH without leaving the origin app's
    # journey: Google sign-in (gms / Chrome custom tabs), permission dialogs. Screens
    # there still record into their own maps, but the navigation cursor stays on the
    # origin app so the round trip becomes one recorded (and replayable) edge.
    transit_packages: list[str] = Field(
        default_factory=lambda: [
            "com.google.android.gms",
            "com.android.chrome",
            "com.android.permissioncontroller",
            "com.google.android.permissioncontroller",
        ]
    )
    # Step labels goto refuses to auto-replay without --allow-destructive (word-boundary
    # match, tap/long-press steps only). Authored flows are exempt by default.
    destructive_labels: list[str] = Field(
        default_factory=lambda: [
            "delete",
            "remove",
            "sign out",
            "log out",
            "logout",
            "pay",
            "buy",
            "purchase",
            "subscribe",
            "unsubscribe",
            "uninstall",
            "format",
            "erase",
            "reset",
            "deactivate",
        ]
    )


def _default_models() -> dict[str, dict[str, Any]]:
    """Shipped, commercially-licensable defaults (PRD §9, §17).

    AGPL OmniParser is present but gated off (``accept_agpl: false``); YOLO has no
    weights; grounding is disabled — so out of the box no AGPL/research/paid component
    runs.
    """
    return {
        # detection
        "yolo": {"weights": None, "device": "mps", "conf": 0.25},
        "omniparser": {"device": "mps", "accept_agpl": False, "box_threshold": 0.05},
        # ocr
        "apple_vision": {"recognition_level": "accurate"},
        "rapidocr": {"lang": "en"},
        "paddleocr": {"lang": "en"},
        "tesseract": {"lang": "eng"},
        "easyocr": {"lang": ["en"]},
        # grounding (referenced only if grounding.enabled)
        "local_vllm": {"base_url": "http://localhost:8000/v1", "model": "Hcompany/Holo1.5-7B"},
        "openai": {
            "model": "gpt-5",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
        },
        "anthropic": {
            "model": "claude-opus-4-8",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com/v1",
        },
        "gemini": {
            "model": "gemini-2.5-flash",
            "api_key_env": "GEMINI_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
        },
        "gemini_flash": {
            "model": "gemini-2.5-flash-lite",
            "api_key_env": "GEMINI_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
        },
    }


class FlagsCfg(BaseModel):
    """Feature-flag deeplink templates (package → URI with ``{query}`` placeholder).

    ``prefs_files`` pins the ``shared_prefs`` XML the read-back verification reads
    (package → filename); unset, every prefs file of the app is searched.

    When a package has either ``prefs_files`` or ``context_keys`` configured, AUA can
    read already-active runtime flags before recording a screen. This keeps maps
    context-aware even when another tool or an earlier app session set the flags.
    """

    model_config = ConfigDict(extra="forbid")
    templates: dict[str, str] = Field(default_factory=dict)
    prefs_files: dict[str, str] = Field(default_factory=dict)
    auto_context: bool = True
    context_keys: dict[str, list[str]] = Field(default_factory=dict)
    context_key_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(?i)(?:^|[_\-.])(experiment|treatment|variant)(?:$|[_\-.])",
            r"(?i)(?:^|[_\-.])flag(?:$|[_\-.])",
        ]
    )
    context_refresh_s: float = 2.0


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_level: str = "warn"
    device: DeviceCfg = Field(default_factory=DeviceCfg)
    perception: PerceptionCfg = Field(default_factory=PerceptionCfg)
    routing: RoutingCfg = Field(default_factory=RoutingCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)
    ocr: OcrCfg = Field(default_factory=OcrCfg)
    detection: DetectionCfg = Field(default_factory=DetectionCfg)
    grounding: GroundingCfg = Field(default_factory=GroundingCfg)
    planner: PlannerCfg = Field(default_factory=PlannerCfg)
    timeouts: TimeoutsCfg = Field(default_factory=TimeoutsCfg)
    models: dict[str, dict[str, Any]] = Field(default_factory=_default_models)
    daemon: DaemonCfg = Field(default_factory=DaemonCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    capture: CaptureCfg = Field(default_factory=CaptureCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    perf: PerfCfg = Field(default_factory=PerfCfg)
    flags: FlagsCfg = Field(default_factory=FlagsCfg)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # -- views -------------------------------------------------------------

    def masked_dict(self) -> dict[str, Any]:
        """Config as a dict safe to print: any secret-ish *value* is masked.

        We never store secrets, but this is belt-and-suspenders so ``config show`` can
        never leak one even if a user pastes a literal key by mistake.
        """
        data = self.model_dump(mode="json")
        _mask_in_place(data)
        return data


def _mask_in_place(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.lower() in _SECRET_KEYS and not k.lower().endswith("_env"):
                obj[k] = "***"
            else:
                _mask_in_place(v)
    elif isinstance(obj, list):
        for item in obj:
            _mask_in_place(item)


# --------------------------------------------------------------------------- helpers


def user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / USER_CONFIG_REL


def find_project_config(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for directory in [cur, *cur.parents]:
        candidate = directory / PROJECT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def read_env_secret(env_name: str | None, env: dict[str, str] | None = None) -> str | None:
    """Read a secret by env-var *name* at runtime. Returns ``None`` if unset/empty."""
    if not env_name:
        return None
    src = env if env is not None else os.environ
    value = src.get(env_name)
    return value or None


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at top level")
    return data


def _coerce_scalar(value: str) -> Any:
    low = value.strip().lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "AUA_SERIAL": ("device", "serial"),
    "AUA_FORMAT": ("output", "format"),
    "AUA_ANNOTATE": ("output", "annotate"),
    "AUA_LOG_LEVEL": ("log_level",),
    "AUA_MAX_TIER": ("routing", "max_tier"),
    "AUA_AUTO_ESCALATE": ("routing", "auto_escalate"),
}


def _set_path(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = tree
    for part in path[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover - defensive
            return
    node[path[-1]] = value


def env_overrides(env: dict[str, str]) -> dict[str, Any]:
    """Build a nested override dict from ``AUA_*`` env vars.

    ``AUA_OCR__CHAIN=apple_vision,rapidocr`` → ``{'ocr': {'chain': [...]}}``.
    ``AUA_OUTPUT__FORMAT=pretty`` → ``{'output': {'format': 'pretty'}}``.
    Plus friendly aliases (``AUA_SERIAL`` …). ``AUA_CONFIG`` / ``AUA_PROFILE`` are
    consumed by the caller, not here.
    """
    out: dict[str, Any] = {}
    for raw_key, raw_val in env.items():
        if not raw_key.startswith("AUA_"):
            continue
        if raw_key in {"AUA_CONFIG", "AUA_PROFILE"}:
            continue
        if raw_key in _ENV_ALIASES:
            path = _ENV_ALIASES[raw_key]
        elif "__" in raw_key:
            parts = raw_key[len("AUA_") :].lower().split("__")
            path = tuple(p for p in parts if p)
        else:
            continue
        if "," in raw_val:
            value: Any = [_coerce_scalar(p) for p in raw_val.split(",")]
        else:
            value = _coerce_scalar(raw_val)
        _set_path(out, path, value)
    return out


def _prune_none(d: dict[str, Any]) -> dict[str, Any]:
    """Drop top-level keys whose value is None (so CLI 'unset' flags don't override)."""
    return {k: v for k, v in d.items() if v is not None}


def load_config(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    profile: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> Config:
    """Load + merge + validate config across all layers (see module docstring)."""
    env = dict(env if env is not None else os.environ)
    profile = profile or env.get("AUA_PROFILE")
    explicit_path = explicit_path or env.get("AUA_CONFIG")

    merged: dict[str, Any] = Config().model_dump(mode="python")

    # File layer: explicit --config replaces discovery; else user then project.
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        merged = _deep_merge(merged, _load_yaml(p))
    else:
        up = user_config_path()
        if up.is_file():
            merged = _deep_merge(merged, _load_yaml(up))
        pp = find_project_config(cwd)
        if pp is not None:
            merged = _deep_merge(merged, _load_yaml(pp))

    # Profile overlay (deep-merge chosen profile over the base).
    if profile:
        profiles = merged.get("profiles", {})
        if profile not in profiles:
            available = ", ".join(sorted(profiles)) or "(none defined)"
            raise ConfigError(
                f"unknown profile '{profile}'", hint=f"Available profiles: {available}."
            )
        merged = _deep_merge(merged, profiles[profile])

    # Environment overrides.
    merged = _deep_merge(merged, env_overrides(env))

    # CLI flag overrides (highest).
    if cli_overrides:
        merged = _deep_merge(merged, _prune_none(cli_overrides))

    try:
        return Config.model_validate(merged)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        msg = first.get("msg", "invalid value")
        raise ConfigError(
            f"invalid config at '{loc}': {msg}",
            hint="Run `aua config show --effective` to inspect the merged config.",
        ) from exc


# --------------------------------------------------------------------------- template


def default_config_yaml() -> str:
    """Commented, license-clean default config for ``aua config init`` (PRD §5, §9)."""
    return """\
# android-ui-analyser configuration (commercially-licensable defaults).
# Secrets are NEVER stored here — reference the env-var NAME (api_key_env) instead.

device:
  serial: null            # null = auto-detect the only/first device
  backend: uiautomator2   # uiautomator2 | accessibility (future)
  compressed_hierarchy: true  # drop non-visible nodes (smaller dumps)

perf:
  prefetch: true              # background hierarchy dump after actions
  predictive_prefetch: true   # also prefetch during settle / from map edges
  async_memory: true          # record map off the analyze critical path
  skip_unchanged_memory: true # skip map write when tree fingerprint unchanged
  reuse_capture_frames: true  # share capture JPEGs with --with-image / settle
  capture_adb_screencap: true # capture loop prefers adb exec-out screencap
  differential: false         # meta.element_diff vs previous analyze
  auto_daemon: true           # CLI auto-starts daemon when enabled but down
  settle_profiles: true       # learn per-action settle budgets
  gate_cache: true            # memoize gate.decide for identical trees

perception:
  gate:
    min_elements: 3
    min_labeled_ratio: 0.15
    vision_packages: ["io.flutter", "com.unity3d", "org.libsdl", "*.WebView"]
    # Element-class matches (e.g. a WebView node) escalate to vision only when the
    # tree is also weak; package/activity matches always escalate.
    soft_min_elements: 8
    soft_min_labeled_ratio: 0.3

routing:
  auto_escalate: true
  max_tier: vision        # text < selector < hierarchy < vision < grounding
  semantic_query_hierarchy_first: true

output:
  format: json            # json | pretty | compact
  annotate: false

ocr:
  enabled: true
  chain: [apple_vision, rapidocr]     # apple_vision is macOS-only; rapidocr is the fallback

detection:
  enabled: true
  chain: [yolo, omniparser]           # yolo (license-clean) first if weights present

grounding:
  enabled: false                      # opt-in; off by default
  chain: [local_vllm, gemini]

planner:
  enabled: false                      # opt-in LLM navigator; also needs --assist / `aua navigate`
  chain: [gemini_flash]

models:
  yolo:         { weights: null, device: mps, conf: 0.25 }   # set weights to enable YOLO
  omniparser:   { device: mps, accept_agpl: false }          # MUST be true to run (AGPL-3.0!)
  apple_vision: { recognition_level: accurate }
  rapidocr:     { lang: en }
  local_vllm:   { base_url: "http://localhost:8000/v1", model: "Hcompany/Holo1.5-7B" }
  openai:       { model: gpt-5, api_key_env: OPENAI_API_KEY }
  anthropic:    { model: claude-opus-4-8, api_key_env: ANTHROPIC_API_KEY }
  gemini:       { model: gemini-2.5-flash, api_key_env: GEMINI_API_KEY }
  gemini_flash: { model: gemini-2.5-flash-lite, api_key_env: GEMINI_API_KEY }

daemon:
  enabled: true
  socket: "~/.cache/android-ui-analyser/daemon.sock"

capture:
  enabled: true           # rolling screencap while daemon is warm
  idle_fps: 2
  burst_fps: 10           # best-effort after actions
  burst_ms: 1500
  extend_burst_on_change: true  # keep bursting while pixels keep changing
  ttl_s: 180
  max_mb: 200
  jpeg_quality: 70
  hint: true              # meta.capture_hint after post-action pixel change
  sidecar: true           # host capture process when full daemon is off

memory:
  enabled: true
  auto_record: true        # record screens + route edges on every analyze/action
  dir: "~/.android-ui-analyser"
  drift_threshold: 0.3     # signature divergence that flags a screen stale
  redact: true             # never store secrets / PII / EditText values verbatim
  suggest: true            # push known_routes/suggested_gotos/map_hint inline into analyze
  suggest_max: 4           # cap on suggested_gotos per analyze
  rank_half_life_days: 3.0 # recency decay for usage-based ranking (days)
  auto_research: true      # surface audit questions to the agent automatically
  research_suggest_max: 3
  # Never the foreground app (excluded from the package vote; never mapped):
  ignore_packages: ["com.android.systemui", "*inputmethod*"]
  # Surfaces a flow passes through (Google auth, permission dialogs) — the journey
  # cursor stays on the origin app so the round trip records as one replayable edge:
  transit_packages: ["com.google.android.gms", "com.android.chrome",
                     "com.android.permissioncontroller", "com.google.android.permissioncontroller"]
  # Step labels goto refuses to auto-replay without --allow-destructive:
  destructive_labels: ["delete", "remove", "sign out", "log out", "logout", "pay", "buy",
                       "purchase", "subscribe", "unsubscribe", "uninstall", "format",
                       "erase", "reset", "deactivate"]

# Feature-flag deeplink templates — required per package (a set-flags scheme is an app's
# private contract, so there are no built-ins). prefs_files is optional: it pins the
# shared_prefs XML `flags set` reads back to verify (default: search all of them).
# flags:
#   auto_context: true
#   templates:
#     com.example.app: "myapp://set-flags?{query}"
#   prefs_files:
#     com.example.app: "flag_overrides.xml"
#   # Optional exact allow-list. Without it, experiment/treatment/variant/flag-like
#   # keys from the configured prefs file are used as the runtime map context.
#   context_keys:
#     com.example.app: [catalog_experiment, services_treatment]

# profiles:
#   cloud:
#     grounding: { enabled: true, chain: [gemini] }
"""
