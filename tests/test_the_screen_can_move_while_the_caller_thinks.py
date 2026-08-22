"""The world can arrive on its own, and settle logic cannot see it — so say so.

Recorded failure: a tap landed on a tab, and then a promotional interstitial appeared **by
itself** and covered the bottom bar. The caller's next tap named a control the interstitial
had covered. It read like a stale-id mistake and was not: the screen moved between two
calls, and nothing the caller did caused it, so no arrival/settle verdict could report it.

`Meta.caller.previous_screen_gone` already answered this question every turn, and was not
delivered where an acting caller reads. Restoring the whole `caller` block would have cost a
measured **199 B (~50 tok) on every action** — `gap_ms`, `ema_ms`, `spread_ms`, `samples`,
`wait_ceiling_*` — telemetry the payload-trimming work removed on purpose. So the *signal*
is delivered rather than the block: `meta.screen_moved` (None-stripped, so an unmoved screen
pays nothing) plus one sentence at the front of the action `note`, which is where a caller
is already reading.

The thing that would make this feature annoying rather than useful is crying wolf, and the
raw fingerprint comparison does exactly that. Measured on one live emulator screen, three
consecutive `analyze` calls with nobody touching the device returned fingerprints
``71e86ef56d9d``, ``71e86ef56d9d``, ``902056fe5693`` and node counts 43, 43, **44** — while
the set of nine actionable ids was byte-identical on all three. A warning keyed on the
fingerprint would have fired on an ordinary back-to-back action. So the verdict is keyed on
what the caller can *act on*: which actionable ids left, and which arrived on top.

Three more ways not to cry wolf, each pinned below: the first call of a session has nothing
to compare, a gap longer than `caller_latency.IDLE_GAP_MS` is someone who walked away rather
than an interstitial, and the caller's *own* action must never be reported as the world
moving — the verdict is therefore decided on the pre-action resolution read, before the
device is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.caller_latency import IDLE_GAP_MS
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ElementNotFoundError
from android_ui_analyser.projection import OBSERVATION_META_PRESETS, Projection
from android_ui_analyser.schema import OutputFormat
from conftest import FakeDevice, make_config

PACKAGE = "com.example.fiction"

_TABS = f"""<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Fixtures" bounds="[40,80][1040,160]"/>
  <node class="android.widget.Button" text="Browse"
        resource-id="{PACKAGE}:id/tab_browse" clickable="true" enabled="true"
        bounds="[40,2200][520,2320]"/>
  <node class="android.widget.Button" text="Saved"
        resource-id="{PACKAGE}:id/tab_saved" clickable="true" enabled="true"
        bounds="[560,2200][1040,2320]"/>
</hierarchy>"""

# The recorded failure: an interstitial arrives on its own, over a bottom bar that is still
# in the tree — so the covered control still resolves and the tap still "succeeds".
_TABS_UNDER_INTERSTITIAL = f"""<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Fixtures" bounds="[40,80][1040,160]"/>
  <node class="android.widget.Button" text="Claim your offer"
        resource-id="{PACKAGE}:id/promo_accept" clickable="true" enabled="true"
        bounds="[40,900][1040,1100]"/>
  <node class="android.widget.Button" text="Browse"
        resource-id="{PACKAGE}:id/tab_browse" clickable="true" enabled="true"
        bounds="[40,2200][520,2320]"/>
  <node class="android.widget.Button" text="Saved"
        resource-id="{PACKAGE}:id/tab_saved" clickable="true" enabled="true"
        bounds="[560,2200][1040,2320]"/>
</hierarchy>"""

# The measured no-wolf case: one more non-actionable node, a different fingerprint, and the
# same nine things you could act on (43 → 44 nodes on the live screen above).
_TABS_PLUS_A_TICKING_LABEL = f"""<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Fixtures" bounds="[40,80][1040,160]"/>
  <node class="android.widget.TextView" text="updated 12:06" bounds="[40,180][1040,240]"/>
  <node class="android.widget.Button" text="Browse"
        resource-id="{PACKAGE}:id/tab_browse" clickable="true" enabled="true"
        bounds="[40,2200][520,2320]"/>
  <node class="android.widget.Button" text="Saved"
        resource-id="{PACKAGE}:id/tab_saved" clickable="true" enabled="true"
        bounds="[560,2200][1040,2320]"/>
</hierarchy>"""

# A wholly different screen: the id the caller is holding is not addressable at all.
_SESSION_EXPIRED = f"""<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Sign in again" bounds="[40,80][1040,160]"/>
  <node class="android.widget.Button" text="Sign in"
        resource-id="{PACKAGE}:id/sign_in" clickable="true" enabled="true"
        bounds="[40,900][1040,1020]"/>
</hierarchy>"""


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    return Engine(make_config(cache={"dir": str(tmp_path)}), device=device)


def _hand_the_caller_a_screen(engine: Engine) -> str:
    """One complete caller turn: observe, stamp what was emitted, then open the next turn."""
    engine.open_caller_turn()
    shown = engine.analyze()
    engine.close_caller_turn(shown.meta.fingerprint)
    engine.open_caller_turn()
    return str(shown.meta.fingerprint)


def _age_the_last_turn(engine: Engine, seconds: float) -> None:
    """Backdate the stamp so this caller's gap classifies as ``idle``."""
    store = engine._caller_latency_store()  # noqa: SLF001 - the record under test
    assert store is not None
    record = json.loads(store.path.read_text(encoding="utf-8"))
    record["ended_at"] = float(record["ended_at"]) - seconds
    store.path.write_text(json.dumps(record), encoding="utf-8")


def _rendered(result: Any) -> str:
    """The action response as the CLI emits it, observation trimmed the default way."""
    from android_ui_analyser.projection import trim_observation_payload

    payload = json.loads(result.render(OutputFormat.compact))
    view = Projection.for_observation(
        "id,text,desc,clickable,enabled,checked,selected,cost", meta="changed"
    )
    return json.dumps(trim_observation_payload(payload, view), separators=(",", ":"))


# ----------------------------------------------------------------- it fires when it should


def test_an_interstitial_that_arrived_on_its_own_is_named_in_the_note(tmp_path: Path) -> None:
    """The recorded failure, end to end: the tap still lands, and the caller is warned.

    The interstitial leaves the bottom bar in the tree, so nothing refuses the action — which
    is precisely why a note is the delivery mechanism. A caller that acts on this response's
    ids recovers; one that keeps using the previous screen's taps the promo.
    """
    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    _hand_the_caller_a_screen(engine)

    device._xml = _TABS_UNDER_INTERSTITIAL  # noqa: SLF001 - arrived by itself
    result = engine.tap(selector={"key": "rid:tab_browse"})

    assert result.ok is True
    assert result.note is not None
    assert result.note.startswith("WARNING:"), result.note
    assert "replaced" in result.note, result.note
    assert result.observation is not None
    assert result.observation.meta.screen_moved, "the machine-readable form has to be there too"


def test_the_signal_survives_an_engine_that_never_opened_a_caller_turn(tmp_path: Path) -> None:
    """The warm daemon's situation, which is where this feature quietly fails to arm.

    A daemon round trip is aua's transport, not a caller, so the daemon's engine deliberately
    never opens a caller turn — the CLI process owns both ends of the gap. The stamp is on
    disk, so the process that reads the screen can still answer the question; reading the
    open turn only would make the whole warning a CLI-only feature.
    """
    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    _hand_the_caller_a_screen(engine)

    device._xml = _TABS_UNDER_INTERSTITIAL  # noqa: SLF001
    engine._caller_turn = None  # noqa: SLF001 - as the daemon serves it
    result = engine.tap(selector={"key": "rid:tab_browse"})

    assert result.observation is not None
    assert result.observation.meta.screen_moved, "the daemon path never armed the comparison"


def test_a_key_miss_says_the_screen_moved_and_hands_back_the_one_that_is_there(
    tmp_path: Path,
) -> None:
    """The other half of the recorded failure: the held id is not addressable any more.

    The resolution read is how AUA knows, so the answer is free — and "your target is gone
    AND the screen changed under you" is recoverable where "re-analyze" is a round trip.
    """
    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    _hand_the_caller_a_screen(engine)

    device._xml = _SESSION_EXPIRED  # noqa: SLF001

    with pytest.raises(ElementNotFoundError) as caught:
        engine.tap(selector={"key": "rid:tab_browse"})

    error = caught.value.to_dict()["error"]
    assert isinstance(error, dict)
    assert error["observation_present"] is True
    meta = error["observation"]["meta"]
    assert meta.get("screen_moved"), "the miss knew the screen had moved and did not say so"


# --------------------------------------------------------------- it stays quiet when it must


def test_an_ordinary_back_to_back_action_says_nothing(tmp_path: Path) -> None:
    """The measured no-wolf case: a new fingerprint, the same set of things to act on.

    Live numbers from the module docstring: 43 → 44 nodes and a changed fingerprint with an
    identical actionable set, with nobody touching the device. A warning keyed on the
    fingerprint fires here, and a caller that is warned every call stops reading warnings.
    """
    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    held = _hand_the_caller_a_screen(engine)

    device._xml = _TABS_PLUS_A_TICKING_LABEL  # noqa: SLF001
    result = engine.tap(selector={"key": "rid:tab_browse"})

    assert result.observation is not None
    assert result.observation.meta.fingerprint != held, "this fixture has to move the fingerprint"
    assert result.observation.meta.screen_moved is None, result.observation.meta.screen_moved
    assert result.note is not None
    assert not result.note.startswith("WARNING:"), result.note


def test_a_healthy_action_adds_no_bytes_at_all(tmp_path: Path) -> None:
    """Zero added cost is the whole reason this is a warning and not a telemetry block."""
    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    _hand_the_caller_a_screen(engine)

    rendered = _rendered(engine.tap(selector={"key": "rid:tab_browse"}))

    assert "screen_moved" not in rendered, rendered


def test_the_first_call_of_a_session_has_nothing_to_compare(tmp_path: Path) -> None:
    """No previous screen means no verdict — not a verdict about a screen nobody was shown."""
    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    engine.open_caller_turn()

    result = engine.tap(selector={"key": "rid:tab_browse"})

    assert result.observation is not None
    assert result.observation.meta.screen_moved is None


def test_a_long_idle_gap_is_a_slow_human_not_an_interstitial(tmp_path: Path) -> None:
    """`gap_ignored` already draws this line for the wait ceiling; the warning follows it.

    A gap past `IDLE_GAP_MS` is someone who walked away, a paused debugger, a session resumed
    the next morning. Of course the screen is different — reporting that as "something arrived
    on its own" is the annoying half of this feature, and it is the same rule the latency
    estimate already refuses to learn from.
    """
    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    engine.open_caller_turn()
    shown = engine.analyze()
    engine.close_caller_turn(shown.meta.fingerprint)
    _age_the_last_turn(engine, (IDLE_GAP_MS / 1000.0) + 60.0)
    engine.open_caller_turn()

    device._xml = _TABS_UNDER_INTERSTITIAL  # noqa: SLF001
    result = engine.tap(selector={"key": "rid:tab_browse"})

    assert result.observation is not None
    assert result.observation.meta.screen_moved is None


# --------------------------------------------------------------------------- delivery shape


def test_the_preset_carries_the_warning_and_not_the_telemetry() -> None:
    """`caller` is 199 B on every action; the one key that matters is free when absent."""
    preset = OBSERVATION_META_PRESETS["changed"]

    assert "screen_moved" in preset
    assert "caller" not in preset


def test_an_absent_warning_is_dropped_from_the_observation() -> None:
    """None-stripped, so the unmoved case pays nothing rather than paying for a null."""
    from android_ui_analyser.schema import AnalyzeResult, Meta, Screen

    payload = AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[],
        meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy"),
    ).as_dict(OutputFormat.json)
    view = Projection.for_observation("id,text", meta="changed")
    assert view is not None

    assert "screen_moved" not in view.apply(payload)["meta"]


def test_the_daemon_boundary_does_not_drop_it(tmp_path: Path) -> None:
    """The warm daemon answers with a plain dict that the CLI validates back into a model.

    `Meta` forbids extra keys, so an undeclared field does not merely go missing — the whole
    rehydration fails and `--format` is silently ignored for that response. This is the same
    boundary that swallowed the attached observation three times.
    """
    from android_ui_analyser.cli import _rehydrate

    device = FakeDevice(hierarchy_xml=_TABS, package=PACKAGE)
    engine = _engine(tmp_path, device)
    _hand_the_caller_a_screen(engine)
    device._xml = _TABS_UNDER_INTERSTITIAL  # noqa: SLF001
    over_the_socket = json.loads(engine.tap(selector={"key": "rid:tab_browse"}).render())

    rebuilt = _rehydrate(over_the_socket)

    assert not isinstance(rebuilt, dict), "the response did not survive validation"
    assert rebuilt.observation.meta.screen_moved
    assert str(rebuilt.note).startswith("WARNING:")
