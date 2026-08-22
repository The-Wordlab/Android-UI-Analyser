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
    # Strategy name registered in ``aua.platforms``. Android remains the only built-in.
    platform: str = "android"
    serial: str | None = None  # null = auto-detect
    backend: str = "uiautomator2"  # uiautomator2 | accessibility (future)
    # Drop non-visible nodes in dump_hierarchy — smaller XML on deep trees.
    compressed_hierarchy: bool = True

    @field_validator("platform")
    @classmethod
    def _normalise_platform(cls, value: str) -> str:
        name = value.strip().lower()
        if not name:
            raise ValueError("must not be empty")
        return name


class HelperCfg(BaseModel):
    """Optional on-device helper (an AccessibilityService APK).

    One switch. Turning ``enabled`` on means "use the helper", and AUA does whatever setup
    that needs: check the target can actually run it, push the bundled APK, switch the
    service on, and confirm it bound. There is no second flag to remember, because a helper
    you have to install by hand is one nobody turns on.

    Off by default and deliberately so. The helper is only a faster answer to questions AUA
    already answers by polling, so an absent, stale or unbindable helper must never change a
    result — it just costs the slower path. Android refuses to bind a sideloaded accessibility
    service unless adbd can run as root, which rules out retail phones and Play-image AVDs;
    on those, setup is skipped once and the run continues on the normal path.
    """

    model_config = ConfigDict(extra="forbid")
    # Use the helper, installing and enabling it if that is what "use" requires.
    enabled: bool = False
    # Escape hatch for "use it if someone already set it up, but never install anything
    # yourself" — an unattended fleet where pushing an APK is somebody else's decision.
    auto_setup: bool = True
    connect_timeout_s: float = 5.0
    # Handing a run to the device costs one fixed handover, and the size of that cost is
    # almost entirely about whether uiautomator2 has already taken the UiAutomation slot.
    # With it attached, the device has to give the slot up and wait for the helper to bind:
    # 2155ms. With nothing attached, the helper is already bound: 16ms. Whole fixed cost
    # 2839ms against 682ms, so the engine resolves the target serial without connecting and
    # only connects afterwards, if the host still needs to.
    #
    # Measured end-to-end through the real executor once it does that:
    #
    #    2 steps   5919ms -> 4670ms   (1.27x)
    #    4 steps   9835ms -> 5618ms   (1.75x)
    #
    # Two is where it starts paying: one step saves about 430ms and cannot cover a 682ms
    # handover, two can. This was 10 while the handover cost 2839ms — which put every flow in
    # this repo below the line, so the feature would have sat switched on and never fired.
    min_flow_steps: int = 2
    # The floor for a run that does *not* start the flow. It is much higher because it is a
    # different trade: by then uiautomator2 is attached, so the slot has to be taken away and
    # handed back around the run, and the host pays both directions. Sharing one floor with
    # the cheap case is what made a mid-flow offload come out slower than not offloading at
    # all. Measured cost of the round trip is recorded next to the value below.
    min_midflow_steps: int = 8
    # Hand runs to the helper even when a warm daemon owns the device. On, because the
    # daemon is how AUA runs by default, so off would mean the helper almost never fires.
    #
    # This is only safe because the outcome is now one of two clean states rather than a
    # spectrum. The engine refuses the handover unless the capture buffer has actually gone
    # quiet, so a run either goes to the device whole or stays on the host entirely; it can
    # no longer stop halfway with the accessibility service pulled out from under it.
    # Measured over ten consecutive runs of a 24-step flow under a warm daemon:
    #
    #     7 runs   24 of 24 steps on device   3.2-4.0s
    #     3 runs   declined, host path        ~19.5s   (17.1s with the helper off)
    #
    # So the win is about 5x and the tax on a refusal is the couple of seconds spent asking.
    # Turn it off to keep a warm daemon strictly on the polling path.
    offload_under_daemon: bool = True


class PerfCfg(BaseModel):
    """Latency knobs — defaults favour the agent hot path (tap → observe → tap)."""

    model_config = ConfigDict(extra="forbid")
    prefetch: bool = True  # background hierarchy dump after actions
    predictive_prefetch: bool = True  # kick prefetch during settle / from map edges
    async_memory: bool = True  # record screens/edges off the analyze critical path
    skip_unchanged_memory: bool = True  # skip map write when tree fingerprint unchanged
    reuse_capture_frames: bool = True  # share capture JPEGs with --with-image / settle
    # Off by default: `adb exec-out screencap -p` makes the *device* encode a full-res PNG,
    # which costs more than u2's JPEG capture even after the host re-encodes it to PNG.
    # Measured end-to-end (ScreenImage out): emulator 720x1280 u2 35ms vs adb 79ms; physical
    # 1440x3120 u2 210ms vs adb 471ms — u2 ~2.2x faster on both. OCR recall over four Settings
    # screens was identical (51/73 recovered either way), so the lossless PNG buys no accuracy.
    # Turn on only when a capture frame must be pixel-exact (lossless diffing, colour checks).
    capture_adb_screencap: bool = False  # capture loop uses adb exec-out screencap
    differential: bool = True  # meta.element_diff vs previous analyze (token-cheap)
    skip_unchanged_analyze: bool = True  # reuse last result when hierarchy XML hash matches
    auto_daemon: bool = True  # CLI auto-starts the warm daemon when enabled but down
    settle_profiles: bool = True  # learn per-action settle budgets from history
    gate_cache: bool = True  # memoize gate.decide for identical tree fingerprints
    # Ceiling on the learned post-action deadline, and on a single learning sample.
    #
    # Both were hard-coded (1600ms / 500ms). They are a deliberate trade, not a bug: the
    # deadline is only *spent* when an action produces no detectable change, so raising it
    # taxes every same-screen tap to help slow ones. Left at the measured-safe defaults and
    # exposed instead, because the right answer for a genuinely slow screen is `--until
    # <predicate>`, which waits on evidence rather than on a blind timer. Raise these only for
    # an app whose transitions are uniformly slow.
    settle_total_max_ms: int = 1600
    settle_learn_cap_ms: float = 500.0
    # -- the arrival extension: bounded truth-completion, not a longer blind wait -------
    #
    # Spent ONLY when the post-action classification says the folded observation is an
    # unready destination (an explicit loading state, or an Activity change that rendered
    # nothing) — i.e. only on calls that would otherwise return a wrong answer. A settled
    # tap never pays it. It re-reads the cheap hierarchy until content arrives, then swaps
    # in the rendered screen, so the agent gets one call with the right answer instead of
    # a stale frame plus a recovery loop.
    #
    # Deliberately a different instrument from `max_wait_ms`. That ceiling caps waits the
    # CALLER sizes — open-ended, and capable of waiting for a change that already happened,
    # which is how a 42s stall was measured. This budget is sized by AUA, waits on specific
    # missing evidence (content that provably has not arrived), and shares the extended
    # confirmation window's ceiling, so the worst post-action wall time does not grow at
    # the default. Raise per-app (auth-heavy screens) toward the 5s backstop the launch
    # content poll already spends; `AUA_ARRIVAL_EXTENSION_MS` sweeps it, 0 disables the
    # wait while keeping the honest verdict.
    arrival_extension_ms: int = 1200

    # -- deliberate settle, and a ceiling the agent cannot lift ----------------
    #
    # Measured on a real device: one `wait-and-analyze --after-change --timeout-ms 45000`
    # blocked for 42.36s, because the change it was told to wait for had already landed
    # while the agent was composing the call. A long ceiling does not buy patience, it buys
    # a stall — the agent is cheaper to re-invoke than to leave blocked, so cap every
    # observation wait here and let it call again. Provisioning waits (install, emulator
    # boot, network shaping) are exempt: they are not observations.
    max_wait_ms: int = 5000
    # -- how the ceiling moves *within* `max_wait_ms` --------------------------
    #
    # `max_wait_ms` above is the hard maximum and the only knob that raises it. These two size
    # the effective ceiling underneath it from what the caller has been measured to cost
    # (`caller_latency`), because 5s is not equally right for every caller: a shell script's
    # re-call costs ~3.9s of tool time and no thinking, so making it hold a 5s wait is the
    # losing trade, while an LLM caller whose think time runs 6-39s is already at the cap and
    # stays there. Adaptive therefore only ever *shortens*; nothing here can exceed
    # `max_wait_ms`.
    #
    # "fixed" opts out of adapting — an adaptive budget makes the same script get a different
    # budget on a different day, which is useless for a measurement run. `AUA_WAIT_CEILING_MS`
    # does the same for one shell loop without editing config. The response reports which mode
    # produced the number, so a reader can tell a reproducible run from an adaptive one.
    wait_ceiling_mode: str = "adaptive"  # "adaptive" | "fixed"
    # Floor, so a zero-latency caller (a shell script, a test) still gets the device's own
    # transition budget: `settle_total_max_ms` alone can spend 1.6s before a screen is settled.
    # Clamped by `max_wait_ms` like everything else — this is a floor under the cap, not a way
    # around it.
    wait_ceiling_min_ms: int = 2000
    # Fixed pause between an action landing and the observation being read. The poll loop
    # can call a one-node splash "settled" after its 45ms quiet window; a deliberate pause
    # is a knob we can sweep for the empty-result/latency trade instead of an emergent
    # property of frame timing. Per action kind, with `default` as the fallback.
    stable_delay_ms: dict[str, int] = Field(
        default_factory=lambda: {
            "default": 250,
            "tap": 250,
            "input": 250,
            "swipe": 350,
            "key": 250,
            # A cold process start draws a splash long before it draws content.
            "launch": 900,
            "open-link": 600,
        }
    )


class LeaseCfg(BaseModel):
    """Per-device leases, so parallel agents stop landing on the same emulator."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # Coordination is host-wide, deliberately independent of ``cache.dir``. Callers use
    # per-run caches to isolate screenshots, journals, proxy rules and session artifacts; if
    # leases followed that override, two agents would each see the same device as free.
    registry_dir: str = "~/.cache/android-ui-analyser"
    # Long waits renew from inside their polling loop; this is the fallback for owners that
    # cannot be bound to a live process.
    ttl_s: int = 120


class TeardownCfg(BaseModel):
    """Undo persistent device changes once nobody holds the device any more.

    A proxied, time-travelled or offline device outlives the agent that made it that way: the
    mutation is a device-global setting, the agent is a process that can be SIGKILLed, and lease
    expiry is lazy — nothing runs at the moment a lease lapses. So the undo is journalled where
    another process can find it, and two nets replay it.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # Sweep other devices' pending undos at the start of a command. A directory glob; the reap
    # itself only happens for devices with no live holder.
    sweep_on_command: bool = True
    # Detached per-device watchdog, so the undo does not depend on anyone running `aua` again.
    watchdog: bool = True
    # How long a change with no live lease is left alone before it is undone. Independent of
    # `lease.ttl_s` on purpose: that number protects a long `--until` wait, this one bounds how
    # long a device may stay dirty.
    grace_s: float = 120.0
    watchdog_poll_s: float = 15.0
    # Stop an aua-started emulator idle this long AND holding no live lease. 0 = off. Covers
    # windowed instances too — an emulator nobody has touched in twenty minutes is nobody's
    # session. Two gates make that safe rather than aggressive: the lease (a working agent renews
    # it every command, and a dead agent's is gone the instant its process is) and the length.
    # "Idle" means no *AUA* activity, so a human clicking a windowed AVD by hand looks idle to us;
    # twenty minutes is long enough that a person at the keyboard never crosses it, while a
    # forgotten emulator crosses it at once. An explicit `--idle-stop 0` is always honoured.
    emulator_idle_stop_s: float = 1200.0


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
    # Let a folded post-action observation escalate to vision when the hierarchy cannot describe
    # the screen. Without it the fold is hierarchy-only, so a canvas/WebView screen returns an
    # empty observation and the caller must spend a second `analyze --source auto` — the exact
    # round trip act-and-observe removes everywhere else. Gated by the same `gate.decide` a normal
    # `analyze` uses, so ordinary screens pay nothing.
    observe_escalates_to_vision: bool = True
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
    #
    # On by default, because the pixels are already in hand. OCR runs alongside the hierarchy
    # (`ocr.augment_hierarchy`) and the visual stable keys are computed from a frame, so every
    # analyze has already captured and decoded a screenshot; `false` only threw it away instead
    # of writing it. What that cost a caller was the one escape hatch out of the element view —
    # `meta.raw_image` came back null and the agent had no frame to look at when the tree could
    # not explain what it was seeing. Writing the PNG we already hold is a few milliseconds, and
    # `_prune_run_frames` bounds the directory, so the old default bought nothing.
    with_image: bool | str = True
    # Columns kept in a folded post-action ``observation``; ``"all"`` restores the full dump.
    #
    # The observation used to be all-or-nothing: the whole screen, or `--no-observe`. Actions
    # carry no `--fields`/`--where-*`, so the only way to get a *cheap* read of the new screen
    # was to disable the observation and run a filtered `analyze` instead. Measured on a
    # 5-scenario run: 37 taps produced 73 separate `analyze` calls and 37 `wait` calls, because
    # one unfilterable call was dearer than two targeted ones. The default path has to be the
    # cheap path, or agents will keep routing around it.
    # `rid` is deliberately absent: `id` is the element's stable identity, so on the screens
    # measured for this change the column restated it on more than half the rows and differed
    # only by the uniqueness ordinal on the rest. It remains a first-class *selector* —
    # `--rid`/`--where-rid` name a class of row without analyzing first — which is a different
    # job from addressing one element, and the flags keep doing it.
    # `cost` is here rather than in `meta`: it is what this control cost last time it was
    # acted on, it is absent unless that control has history, and the `changed` meta preset
    # below deliberately drops `slow_controls`. Without the column a learned cost cannot
    # reach an acting caller at all.
    observation_fields: str = "id,text,desc,clickable,enabled,checked,selected,cost"
    # `meta` keys kept in that observation: a preset name from
    # `projection.OBSERVATION_META_PRESETS`, `"all"`, or an explicit comma-separated list.
    #
    # The second half of the same argument. Trimming the element rows left the full `meta`
    # block riding along on every action — measured at 299 of 919 tokens on one real screen,
    # 16 of its keys empty and the largest three (`research_tasks`, `suggested_deeplinks`,
    # `capture_hint`) answering a question no action asked. `changed` keeps what an action
    # does raise: did the screen move, where am I, and anything a caller must not miss.
    # Deliberately a separate dial from `observation_fields`, because wanting every column is
    # not the same as wanting every hint, and one knob could not express that.
    observation_meta: str = "changed"
    # Emit the derived `next_actions` list on every action response. Off, and the default is
    # measured: on one real journalled response the list was 1384 bytes / 346 tokens — 25% of
    # the whole response, and more than the entire 1301-byte `elements` list it was a filtered
    # subset of. It existed to save an agent from scanning ~50 observation nodes for the
    # actionable ones; the observation is now trimmed to ~20 rows with `clickable` on each, so
    # that scan is a one-line filter over a list the caller already holds. Turn it on if you
    # want the pre-filtered form; `Element.cost` carries the learned per-control cost either
    # way, which was the only thing the list said that `elements` could not.
    next_actions: bool = False


class _ChainCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    chain: list[str] = Field(default_factory=list)


class OcrCfg(_ChainCfg):
    enabled: bool = True
    chain: list[str] = Field(default_factory=lambda: ["apple_vision", "rapidocr"])
    # On macOS, overlap Apple Vision with hierarchy capture and fuse both observations.
    augment_hierarchy: bool = True
    # Withhold OCR readings of text the hierarchy already reports. Measured on one app
    # screen: 14 of 16 readings were pure duplication, one was the clock, and one was a
    # misread of a label the tree had right. Set false to see the unfiltered OCR pass.
    drop_redundant: bool = True


class DetectionCfg(_ChainCfg):
    enabled: bool = True
    chain: list[str] = Field(default_factory=lambda: ["yolo", "omniparser"])


class GroundingCfg(_ChainCfg):
    enabled: bool = False  # opt-in (PRD §7.2)
    chain: list[str] = Field(default_factory=lambda: ["local_vllm", "gemini", "openai"])


class PlannerCfg(_ChainCfg):
    """Optional LLM navigator for goto/flow recovery + `aua navigate` (PRD §7.3)."""

    enabled: bool = False  # opt-in; also needs a per-call --assist (or `aua navigate`)
    chain: list[str] = Field(default_factory=lambda: ["gemini_flash"])


class PolicyCfg(_ChainCfg):
    """Optional guarded next-call selection; never executes model output."""

    enabled: bool = False
    chain: list[str] = Field(default_factory=lambda: ["functiongemma"])
    mode: Literal["off", "shadow", "advisory"] = "off"
    strategy: Literal["single", "selective_hybrid"] = "single"
    primary_reviews: int = Field(default=2, ge=1, le=5)
    reviewer_reviews: int = Field(default=3, ge=1, le=5)
    candidate_scope: Literal["goal_matched", "safe_visible"] = "goal_matched"

    @field_validator("mode", mode="before")
    @classmethod
    def _mode_survives_the_env_round_trip(cls, v: Any) -> Any:
        """Accept the boolean ``False`` that ``off`` becomes on its way through the env layer.

        A detached daemon receives this slice as ``AUA_POLICY__MODE=off``, and
        :func:`_coerce_scalar` maps the string ``"off"`` to ``False`` like any other
        off/on flag. Validation then rejected ``False`` for this Literal, so the daemon
        child died at startup on the DEFAULT config — silently, because only
        ``daemon.log`` records the traceback. ``off`` is the only member of this Literal
        that collides with that coercion, so mapping ``False`` back is unambiguous.
        """
        return "off" if v is False else v

    # Four is the widest choice set any adapter in this line was trained and evaluated on.
    # A given adapter may authenticate fewer; its manifest is what the provider enforces.
    max_candidates: int = Field(default=4, ge=1, le=4)


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
    # When set (>0), daemon also serves screen-changed WebSocket push on 127.0.0.1:port.
    push_ws_port: int = 0
    # Poll interval for the push / wait_changed fingerprint watcher (host-side).
    watch_interval_ms: int = 150
    # Exit after this long with no client request (0 = never). Agent sessions end without
    # stopping their daemon, and every survivor keeps polling a device forever.
    idle_ttl_s: int = 1800


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
    # Pause sampling after this long with no client request (0 = never). Frames already on
    # disk stay readable, so `capture last` still works minutes after the agent went quiet;
    # only the screencap load stops. The next request resumes it.
    idle_pause_s: int = 120


class LogsCfg(BaseModel):
    """What the app logged during an action, folded into that action's observation.

    Defaults are measured, not chosen. On one real app: an idle two-second window logged 0
    lines, an ordinary tap 11 (all framework noise, 0 after filtering), and a cold launch 210
    (~30 KB) of which every one of the 113 ``I`` lines came from a third-party SDK or the ART
    runtime. Hence a level *set* rather than a floor — ``I`` is noisier than ``D`` on Android,
    so a floor keeps the wrong half — and a per-tag cap so one chatty logger cannot spend the
    whole budget.

    Every field here is one setting for **every** app on the host. Per-app overrides are
    deliberately not config: they live beside that app's map under ``memory.dir``
    (:class:`~android_ui_analyser.memory.AppLogPrefs`), so the preference follows the app
    instead of the project, and an app's own tag names never have to be written into a config
    file somebody might commit.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # Priority set, not a floor. `F` is added back however narrow this is: a caller must not be
    # able to hide the line that explains a crash.
    levels: str = "DWEF"
    limit: int = 20  # lines attached, head+tail when the window overflows
    per_tag: int = 5  # lines any single tag may contribute before it is capped
    scan_lines: int = 600  # bounded read from one already-short last-action window
    # Extra tags to drop on top of the built-in generic framework/SDK list. The place to name an
    # app's own chatty logger — that belongs in a user's config, never in this repository.
    deny_tags: list[str] = Field(default_factory=list)
    # Tags that must survive the deny list: the way to read a library the built-in list hides by
    # default, without giving up the rest of the filter.
    keep_tags: list[str] = Field(default_factory=list)
    # When non-empty, the ONLY tags folded in — for a caller that knows which logger it is
    # chasing. `F` still survives, so a narrow filter can never hide a crash.
    only_tags: list[str] = Field(default_factory=list)


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
        # Keep accurate recognition, but reduce pixel work. Boxes are mapped back to original
        # screen coordinates, and 720px retained the visible labels in the benchmark fixture.
        "apple_vision": {"recognition_level": "accurate", "max_width": 720},
        "rapidocr": {"lang": "en"},
        "paddleocr": {"lang": "en"},
        "tesseract": {"lang": "eng"},
        "easyocr": {"lang": ["en"]},
        # grounding (referenced only if grounding.enabled)
        "local_vllm": {"base_url": "http://localhost:8000/v1", "model": "Hcompany/Holo1.5-7B"},
        "openai": {
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "reasoning_effort": "none",
            "screen_image_detail": "high",
            "screen_preview_max_width": 720,
            "screen_preview_jpeg_quality": 55,
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
        # Optional local-only policy selector. The base model must be an existing absolute
        # directory. A null adapter uses AUA's small bundled LoRA; an explicit adapter must be
        # an existing absolute directory. The provider never resolves or downloads a repo ID.
        "functiongemma": {
            "model_path": None,
            "adapter_path": None,
            "max_tokens": 24,
            "model_sha256": None,
            "adapter_sha256": None,
            "manifest_sha256": None,
        },
        # Optional larger text-only semantic reviewer. The model must already exist in an
        # absolute local directory; AUA never downloads it. Advisory use is an explicit local
        # operator choice and remains behind the policy guard.
        "gemma4": {
            "model_path": None,
            "revision": None,
            "max_tokens": 512,
            "max_mode": "shadow",
        },
        # Optional text-only Qwen3 selector. Like the others the model must already exist in an
        # absolute local directory; AUA never downloads it. ``adapter_path`` points at a LoRA
        # directory containing adapters.safetensors. Qwen3's chat template has no developer role,
        # so the provider renders the activation on a system turn to match how it was trained.
        "qwen3": {
            "model_path": None,
            "adapter_path": None,
            "max_tokens": 48,
            "max_mode": "shadow",
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


class DashboardCfg(BaseModel):
    """Defaults for ``aua dashboard``: reachable at ``http://aua.local/`` out of the box.

    ``aua dashboard start`` publishes an mDNS name, binds every interface, and serves
    without a token, so the dashboard is something you type rather than something you
    scan. This is a deliberate default for a tool that is used on a developer's own
    network, and it is worth being plain about what it means: anything that can reach the
    port gets the whole dashboard, which drives the device, streams logcat, and queries
    app databases. Every start says so in its output.

    Each field is overridable in config and on the command line, and the flag wins:

    * ``--auth`` re-arms the access token for one start;
    * ``--local`` pulls the dashboard back to loopback only;
    * ``--name ""`` serves without publishing a name;
    * ``--port N`` pins an exact port.

    Set ``auth: true`` (or ``lan: false``) in your config to make the guarded shape your
    own default on a network you do not control.
    """

    model_config = ConfigDict(extra="forbid")
    # mDNS hostname to publish, e.g. "aua" serves http://aua.local/. Implies ``lan``.
    # Empty or null publishes nothing and leaves the dashboard on its IP.
    name: str | None = "aua"
    # Bind every interface rather than loopback only.
    lan: bool = True
    # Require the access token whenever the dashboard is network-bound. Off by default:
    # turning it on costs one token-bearing URL per browser and survives restarts poorly,
    # which is the friction the published name exists to remove.
    auth: bool = False
    # Exact port. Unset means 80 when a name is published, else 48765. Port 80 needs no
    # privilege on macOS but does on Linux, so an unbindable 80 falls back rather than
    # failing the start.
    port: int | None = None


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
    policy: PolicyCfg = Field(default_factory=PolicyCfg)
    timeouts: TimeoutsCfg = Field(default_factory=TimeoutsCfg)
    models: dict[str, dict[str, Any]] = Field(default_factory=_default_models)
    daemon: DaemonCfg = Field(default_factory=DaemonCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    capture: CaptureCfg = Field(default_factory=CaptureCfg)
    dashboard: DashboardCfg = Field(default_factory=DashboardCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    logs: LogsCfg = Field(default_factory=LogsCfg)
    perf: PerfCfg = Field(default_factory=PerfCfg)
    helper: HelperCfg = Field(default_factory=HelperCfg)
    lease: LeaseCfg = Field(default_factory=LeaseCfg)
    teardown: TeardownCfg = Field(default_factory=TeardownCfg)
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
    "AUA_PLATFORM": ("device", "platform"),
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
        if path[-1] in {"chain", "destructive_labels"}:
            # These settings are lists even when they contain one value. Treating a one-item
            # value as a scalar breaks detached-daemon transport for both the default policy
            # chain and a caller's narrow destructive-action safety lexicon.
            value: Any = [_coerce_scalar(p) for p in raw_val.split(",") if p.strip()]
        elif "," in raw_val:
            value = [_coerce_scalar(p) for p in raw_val.split(",")]
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

helper:                       # optional on-device helper APK (AccessibilityService)
  enabled: false              # use it when installed + bound; polling stays the fallback

  connect_timeout_s: 5.0
  min_flow_steps: 2           # one step cannot repay the handover; two can

perf:
  prefetch: true              # background hierarchy dump after actions
  predictive_prefetch: true   # also prefetch during settle / from map edges
  async_memory: true          # record map off the analyze critical path
  skip_unchanged_memory: true # skip map write when tree fingerprint unchanged
  reuse_capture_frames: true  # share capture JPEGs with --with-image / settle
  capture_adb_screencap: false # capture via adb screencap (lossless PNG, ~2.2x slower)
  differential: true          # meta.element_diff vs previous analyze
  skip_unchanged_analyze: true  # skip re-parse when hierarchy XML hash matches
  auto_daemon: true           # CLI auto-starts daemon when enabled but down
  settle_profiles: true       # learn per-action settle budgets
  arrival_extension_ms: 1200  # bounded in-action wait when the destination is provably unready
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
  with_image: true        # save the frame each analyze already captured (see meta.raw_image)
  # The post-action `observation` budget — two independent dials, either accepting "all".
  observation_fields: id,text,desc,clickable,enabled,checked,selected,cost
  observation_meta: changed   # changed | all | <comma-separated meta keys>
  next_actions: false         # also emit the derived pre-filtered actionable list

ocr:
  enabled: true
  chain: [apple_vision, rapidocr]     # apple_vision is macOS-only; rapidocr is the fallback
  augment_hierarchy: true             # run Apple OCR alongside every hierarchy observation

detection:
  enabled: true
  chain: [yolo, omniparser]           # yolo (license-clean) first if weights present

grounding:
  enabled: false                      # opt-in; off by default
  # Ordered fallback: providers without configured API keys are skipped automatically.
  chain: [local_vllm, gemini, openai]

planner:
  enabled: false                      # opt-in LLM navigator; also needs --assist / `aua navigate`
  chain: [gemini_flash]

policy:
  enabled: false                      # optional local selector; execution needs explicit session autopilot
  mode: "off"                         # off | shadow (metrics only) | advisory (returns exact call)
  chain: [functiongemma]
  strategy: single                    # single | selective_hybrid (fast primary + semantic fallback)
  primary_reviews: 2                  # counterfactual primary votes in selective_hybrid mode
  reviewer_reviews: 3                 # unanimous fallback votes required before advice
  candidate_scope: goal_matched       # safe_visible is for guarded hybrid navigation trials
  max_candidates: 4                  # guard-approved candidates visible to the model

models:
  yolo:         { weights: null, device: mps, conf: 0.25 }   # set weights to enable YOLO
  omniparser:   { device: mps, accept_agpl: false }          # MUST be true to run (AGPL-3.0!)
  apple_vision: { recognition_level: accurate, max_width: 720 }  # downscale, keep coordinates
  rapidocr:     { lang: en }
  local_vllm:   { base_url: "http://localhost:8000/v1", model: "Hcompany/Holo1.5-7B" }
  openai:       { model: gpt-5.6-luna, api_key_env: OPENAI_API_KEY, reasoning_effort: none,
                  screen_image_detail: high, screen_preview_max_width: 720,
                  screen_preview_jpeg_quality: 55 }
  anthropic:    { model: claude-opus-4-8, api_key_env: ANTHROPIC_API_KEY }
  gemini:       { model: gemini-2.5-flash, api_key_env: GEMINI_API_KEY }
  gemini_flash: { model: gemini-2.5-flash-lite, api_key_env: GEMINI_API_KEY }
  # Base model is local/external. null (or "bundled") uses AUA's small packaged LoRA adapter.
  functiongemma: { model_path: null, adapter_path: null, max_tokens: 24,
                   model_sha256: null, adapter_sha256: null, manifest_sha256: null }
  gemma4:       { model_path: null, revision: null, max_tokens: 512, max_mode: shadow }
  qwen3:        { model_path: null, adapter_path: null, max_tokens: 48, max_mode: shadow }

daemon:
  enabled: true
  socket: "~/.cache/android-ui-analyser/daemon.sock"
  # Per-serial sockets: when device.serial / --serial is set, the live path is
  # ``<socket>.<sanitized-serial>`` so multiple warm daemons can coexist.
  push_ws_port: 0         # >0 → localhost WebSocket push of screen_changed events
  watch_interval_ms: 150  # host fingerprint poll for wait_changed / push

lease:
  enabled: true
  registry_dir: "~/.cache/android-ui-analyser" # host-wide; never follows per-run cache.dir
  ttl_s: 120              # fallback only for legacy owners without a live process identity

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
