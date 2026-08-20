"""A setup flow can put an app into a state its UI cannot reach: writing shared preferences.

The gap this closes. Switching a debuggable build's backend, marking onboarding as already
seen, or turning on a persisted developer toggle is *state*, not navigation — there is often no
screen for it at all. Flows could already apply feature flags (a deeplink the app must declare)
and mutate SQLite; a plain ``shared_prefs/<file>.xml`` had no surface, even though the device
layer has read those files over ``run-as`` for as long as ``flags set`` has verified itself
against them. Writing one was the missing half of a read AUA already had.

Both observed failures are in here. A deeplink that writes preferences returns as soon as the
intent is delivered while the app flushes on a background thread, so a ``stop_app`` right after
kills the process first and the preference is silently lost with every step reporting OK; and
some state (an environment held in a DataStore protobuf) has no flow step at all, so setup had
to walk a developer screen.

What the tests pin, and why each one is not obvious:

- **Round-trip.** ``check_saveable`` re-parses a flow's own rendering, so a key that renders
  away is silently dropped by the very check that proves a flow loads. ``values:`` carries
  YAML scalars whose *types* select the Android element (``<boolean>``/``<int>``/``<string>``),
  so a lossy render is a behaviour change, not a cosmetic one.
- **Destructive by kind.** No label can prove a prefs write safe and no missing label can make
  it look safe, so the destructive lexicon — which reads tap labels — cannot classify it.
  ``goto`` learns routes by observation and replays them speculatively; a step that rewrites
  persisted app data must never be replayed to satisfy a navigation goal.
- **The app is stopped first.** A live process holds its own copy of every SharedPreferences
  file and writes it back on exit, so a write into the XML underneath a running app is reverted
  the moment the app is backgrounded — the write "succeeds" and then vanishes. That makes the
  force-stop part of the contract, not an optimisation, and it is asserted as an ordering.
- **Write-ahead undo.** The preference outlives the agent that set it. A build left pointing at
  a staging backend is inherited by whoever picks the device up next, so the previous file is
  journalled *before* it is replaced and ``aua teardown`` puts it back.
- **Parse-time refusal.** A malformed step must fail before the device is touched: half-applied
  setup is worse than no setup, and the flow author gets the error next to the YAML.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import device_ledger
from android_ui_analyser import flags as flags_mod
from android_ui_analyser.config import Config
from android_ui_analyser.device import Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import (
    _ARG_ALIAS,
    _KEYS,
    _KINDS,
    check_saveable,
    parse_flow_yaml,
    render_flow_yaml,
)
from android_ui_analyser.memory import (
    INHERENTLY_DESTRUCTIVE_KINDS,
    RouteStep,
    is_destructive_step,
    route_step_risks,
    step_display,
)
from android_ui_analyser.platforms import (
    CAPABILITY_METHODS,
    NormalizedTree,
    PlatformAdapter,
    register_platform,
)
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.services import FEATURE_FLAGS
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import DeviceInfo
from conftest import FakeDevice, make_config

PKG = "com.example.app"
PREFS_FILE = "example_settings.xml"
HEAD = "schema_version: 1\nname: switch_backend\napp: com.example.app\nsteps:\n"

STEP_YAML = (
    "  - prefs_write:\n"
    f"      file: {PREFS_FILE}\n"
    "      values:\n"
    "        backend_env: staging\n"
    "        onboarding_seen: true\n"
    "        retry_budget: 3\n"
)


def _flow(step_yaml: str = STEP_YAML):
    return parse_flow_yaml(HEAD + step_yaml, name="switch_backend")


def _one(step_yaml: str = STEP_YAML) -> RouteStep:
    return _flow(step_yaml).steps[0]


def _engine(tmp_path: Path, device: Device) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perf={"async_memory": False},
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _device(prefs: dict[str, str] | None = None) -> FakeDevice:
    return FakeDevice(package=PKG, prefs={PREFS_FILE: dict(prefs or {})} if prefs else {})


# --------------------------------------------------------------------------- parsing


def test_the_step_parses_into_a_file_and_typed_values() -> None:
    step = _one()

    assert step.kind == "prefs-write"
    assert step.arg == PREFS_FILE
    assert step.data["values"] == {
        "backend_env": "staging",
        "onboarding_seen": True,
        "retry_budget": 3,
    }


def test_the_yaml_key_and_argument_alias_follow_the_other_kinds() -> None:
    """A step nobody can spell is a step nobody uses."""
    assert _KINDS["prefs_write"] == "prefs-write"
    assert _KEYS["prefs-write"] == "prefs_write"
    assert _ARG_ALIAS["prefs-write"] == "file"


def test_a_bare_file_name_gains_the_extension_android_writes() -> None:
    """`getSharedPreferences("example_settings")` is the file `example_settings.xml`."""
    assert _one(STEP_YAML.replace(PREFS_FILE, "example_settings")).arg == PREFS_FILE


def test_the_step_round_trips_through_yaml() -> None:
    flow = _flow()
    rendered = render_flow_yaml(flow)

    assert "prefs_write:" in rendered
    assert parse_flow_yaml(rendered, name="switch_backend").steps == flow.steps
    assert check_saveable(flow) == [], "a prefs_write flow must be saveable"


def test_an_explicit_relaunch_choice_round_trips() -> None:
    """Rendered away, `relaunch: false` would silently start the app the flow left dead."""
    flow = _flow(STEP_YAML + "      relaunch: false\n")

    assert flow.steps[0].data["relaunch"] is False
    rendered = render_flow_yaml(flow)
    assert "relaunch: false" in rendered
    assert parse_flow_yaml(rendered, name="switch_backend").steps == flow.steps


def test_a_parameterized_value_resolves_like_every_other_flow_value() -> None:
    """The point of the step is switching an environment, which is exactly what params are for."""
    from android_ui_analyser.flows import resolve_params

    flow = parse_flow_yaml(
        "schema_version: 1\nname: switch_backend\nparams:\n  ENV: ''\nsteps:\n"
        f"  - prefs_write: {{file: {PREFS_FILE}, values: {{backend_env: '${{ENV}}'}}}}\n",
        name="switch_backend",
    )

    steps = resolve_params(flow, {"ENV": "staging"})

    assert steps[0].data["values"] == {"backend_env": "staging"}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (f"  - prefs_write: {PREFS_FILE}\n", "mapping"),
        (f"  - prefs_write: {{file: {PREFS_FILE}}}\n", "values"),
        (f"  - prefs_write: {{file: {PREFS_FILE}, values: {{}}}}\n", "values"),
        ("  - prefs_write: {values: {a: b}}\n", "file"),
        (f"  - prefs_write: {{file: {PREFS_FILE}, values: {{a: [1, 2]}}}}\n", "a"),
        (f"  - prefs_write: {{file: {PREFS_FILE}, values: {{a: null}}}}\n", "a"),
        (f"  - prefs_write: {{file: ../other/{PREFS_FILE}, values: {{a: b}}}}\n", "file"),
        (
            f"  - prefs_write: {{file: {PREFS_FILE}, values: {{a: b}}, relaunch: yes please}}\n",
            "relaunch",
        ),
        (f"  - prefs_write: {{file: {PREFS_FILE}, values: {{a: b}}, nope: 1}}\n", "nope"),
    ],
)
def test_a_malformed_step_is_refused_at_parse_time(body: str, expected: str) -> None:
    """Before a device is touched: half-applied setup is worse than none."""
    with pytest.raises(UsageError) as err:
        parse_flow_yaml(HEAD + body, name="switch_backend")
    assert expected in str(err.value)


def test_a_nonfinite_float_is_refused_before_the_device_is_touched() -> None:
    with pytest.raises(UsageError, match="finite"):
        parse_flow_yaml(
            HEAD + f"  - prefs_write: {{file: {PREFS_FILE}, values: {{ratio: .nan}}}}\n",
            name="switch_backend",
        )


# --------------------------------------------------------------------------- destructiveness


def test_the_kind_is_destructive_on_its_own() -> None:
    """An empty lexicon must not make it look safe — there is no label to match."""
    assert "prefs-write" in INHERENTLY_DESTRUCTIVE_KINDS
    assert is_destructive_step(_one(), []) is True


def test_a_learned_route_may_never_replay_it_to_reach_a_screen() -> None:
    risks = route_step_risks(_one(), origin_package=PKG, destructive_labels=[])
    codes = {risk["code"] for risk in risks}

    assert "destructive" in codes
    assert codes - {"destructive"}, (
        "a goto must also require --allow-unsafe, not only a label opt-in"
    )


def test_a_dry_run_shows_which_file_it_rewrites() -> None:
    """`flow run --dry-run` is the last review before persisted app data is overwritten."""
    display = step_display(_one())

    assert PREFS_FILE in display
    assert "prefs-write" in display


def test_the_executor_refuses_the_step_without_the_destructive_opt_in(tmp_path: Path) -> None:
    device = _device()
    engine = _engine(tmp_path, device)

    failure, _res = engine._run_steps(_flow().steps, origin_package=PKG, allow_destructive=False)

    assert failure is not None
    assert failure.code == "destructive_step"
    assert not [call for call in device.calls if call[0] == "write_app_file"]


# --------------------------------------------------------------------------- the write itself


def test_the_write_reaches_the_prefs_the_device_serves_back(tmp_path: Path) -> None:
    """Written through the real code path, read back through the real read path."""
    device = _device({"backend_env": "production", "keep_me": "yes"})
    engine = _engine(tmp_path, device)

    failure, _res = engine._run_steps(_flow().steps, origin_package=PKG, allow_destructive=True)

    assert failure is None
    read = flags_mod.read_prefs(
        device,
        PKG,
        {"backend_env": "staging", "onboarding_seen": "true", "retry_budget": "3"},
        prefs_file=PREFS_FILE,
    )
    assert read.verified is True
    assert read.ignored == []
    assert read.mismatched == {}
    assert device.prefs[PREFS_FILE]["keep_me"] == "yes", "an unrelated entry must survive"


def test_the_written_xml_is_the_shape_android_writes(tmp_path: Path) -> None:
    """Types are not cosmetic: the app reads `getBoolean`, and a string there throws."""
    device = _device()
    engine = _engine(tmp_path, device)

    engine.prefs_write(
        PKG,
        PREFS_FILE,
        {"backend_env": "staging", "onboarding_seen": True, "retry_budget": 3},
        relaunch=False,
    )

    xml = device.read_app_file(PKG, f"shared_prefs/{PREFS_FILE}").decode("utf-8")
    assert '<string name="backend_env">staging</string>' in xml
    assert '<boolean name="onboarding_seen" value="true" />' in xml
    assert '<int name="retry_budget" value="3" />' in xml
    assert xml.startswith("<?xml")


def test_an_authored_flow_runs_the_step_while_goto_still_refuses_it(tmp_path: Path) -> None:
    """The split that makes the step usable: authored intent proceeds, learned replay does not.

    `flow run` allows destructive steps by default because the author wrote them down; the
    same step reached through a learned route is refused. Both halves are asserted here so a
    later change to either default cannot quietly move the line.
    """
    flow_file = tmp_path / "switch_backend.yaml"
    flow_file.write_text(HEAD + STEP_YAML, encoding="utf-8")
    device = _device()
    engine = _engine(tmp_path, device)

    ran = engine.flow_run(file=str(flow_file))

    assert ran["ok"] is True
    assert device.prefs[PREFS_FILE]["backend_env"] == "staging"

    blocked = _engine(tmp_path, _device()).flow_run(file=str(flow_file), allow_destructive=False)
    assert blocked["ok"] is False
    assert blocked["code"] == "destructive_step"


def test_a_first_ever_preference_creates_the_directory_the_app_never_made(tmp_path: Path) -> None:
    """Pre-seeding a fresh install is the main use, and a fresh install has no `shared_prefs`."""
    device = _device()
    result = _engine(tmp_path, device).prefs_write(
        PKG, PREFS_FILE, {"onboarding_seen": True}, relaunch=False
    )

    assert result["created"] is True
    assert any(
        "mkdir" in call[1][0] and "shared_prefs" in call[1][0]
        for call in device.calls
        if call[0] == "shell"
    )


def test_an_entry_the_writer_cannot_model_is_carried_through_untouched() -> None:
    """A `<set>` is real prefs content this module has no vocabulary for.

    Rebuilding the file from the requested keys would delete it, and the flow author would
    have no way to know: the app just starts behaving as though the user had never chosen
    anything. The merge parses, so anything unrecognised survives.
    """
    original = (
        "<?xml version='1.0' encoding='utf-8' standalone='yes' ?>\n<map>\n"
        '    <set name="chosen_topics">\n        <string>alpha</string>\n    </set>\n'
        '    <string name="backend_env">production</string>\n</map>'
    )

    merged = flags_mod.merge_prefs_xml(original, {"backend_env": "staging"})

    assert "<set" in merged and "<string>alpha</string>" in merged
    assert '<string name="backend_env">staging</string>' in merged
    assert "production" not in merged


def test_prefs_xml_that_does_not_parse_is_refused_rather_than_replaced() -> None:
    """The file holds app state AUA did not put there; a clobber is unrecoverable."""
    with pytest.raises(UsageError) as err:
        flags_mod.merge_prefs_xml("<map><string name=", {"a": "b"})

    assert "does not parse" in str(err.value)


def test_the_app_is_stopped_before_the_file_is_replaced(tmp_path: Path) -> None:
    """A live process writes its in-memory copy back on exit and reverts the file."""
    device = _device()
    engine = _engine(tmp_path, device)

    engine.prefs_write(PKG, PREFS_FILE, {"backend_env": "staging"})

    names = [call[0] for call in device.calls]
    assert names.index("stop_app") < names.index("shell"), "the pre-stop listing can be stale"
    assert names.index("stop_app") < names.index("write_app_file")


def test_the_app_comes_back_by_default_and_stays_down_when_asked(tmp_path: Path) -> None:
    """Cold start is when the app re-reads the file, so relaunching is the useful default."""
    device = _device()
    engine = _engine(tmp_path, device)

    result = engine.prefs_write(PKG, PREFS_FILE, {"backend_env": "staging"})
    assert result["relaunched"] is True
    assert "launch_app" in [call[0] for call in device.calls]

    quiet = _device()
    _engine(tmp_path, quiet).prefs_write(PKG, PREFS_FILE, {"a": "b"}, relaunch=False)
    assert quiet.prefs[PREFS_FILE]["a"] == "b"
    assert "launch_app" not in [call[0] for call in quiet.calls]


def test_a_non_debuggable_build_fails_loudly_instead_of_claiming_success(tmp_path: Path) -> None:
    """`run-as` refuses production builds by printing, not by exiting non-zero."""
    device = FakeDevice(package=PKG, run_as_error="run-as: package not debuggable")
    engine = _engine(tmp_path, device)

    with pytest.raises(Exception) as err:
        engine.prefs_write(PKG, PREFS_FILE, {"backend_env": "staging"}, relaunch=False)

    assert "debuggable" in str(err.value)
    assert not [call for call in device.calls if call[0] == "write_app_file"]


# --------------------------------------------------------------------------- write-ahead undo


def test_the_previous_file_is_journalled_before_it_is_replaced(tmp_path: Path) -> None:
    """The preference outlives this process, so the undo has to exist before the write does.

    Recorded *before* ``write_app_file``, not after: a crash between the two leaves a redundant
    undo, which is harmless, while the other order leaves a build silently pointed at another
    test's backend and nothing that knows how to put it back.
    """
    device = _device({"backend_env": "production"})
    engine = _engine(tmp_path, device)
    written: list[str] = []
    real_write = device.write_app_file

    def spy(package: str, path: str, data: bytes) -> None:
        written.append(path)
        real_write(package, path, data)

    device.write_app_file = spy  # type: ignore[method-assign]
    recorded_when_written: list[list[device_ledger.Entry]] = []
    real_record = engine.record_device_change

    def record_spy(**kwargs: Any) -> None:
        real_record(**kwargs)
        recorded_when_written.append(list(written))

    engine.record_device_change = record_spy  # type: ignore[method-assign]

    engine.prefs_write(PKG, PREFS_FILE, {"backend_env": "staging"}, relaunch=False)

    assert recorded_when_written == [[]], "the undo was journalled after the write"
    entries = device_ledger.read_ledger(device.serial)
    assert [e.kind for e in entries] == ["app_prefs"]
    assert entries[0].op == "restore_app_prefs"
    assert Path(entries[0].args["backup_path"]).is_file()


def test_the_registered_undo_puts_the_previous_preferences_back(tmp_path: Path) -> None:
    """A recorded undo nobody can replay is theatre, so replay it against the same device."""
    device = _device({"backend_env": "production", "keep_me": "yes"})
    engine = _engine(tmp_path, device)

    engine.prefs_write(PKG, PREFS_FILE, {"backend_env": "staging"}, relaunch=False)
    assert device.prefs[PREFS_FILE]["backend_env"] == "staging"

    outcome = device_ledger.replay(
        device.serial,
        context=device_ledger.UndoContext(
            serial=device.serial,
            device=device,
            capability=lambda name: flags_mod if name == FEATURE_FLAGS else None,
        ),
    )

    assert not outcome["failed"], outcome
    assert device.prefs[PREFS_FILE]["backend_env"] == "production"
    assert device.prefs[PREFS_FILE]["keep_me"] == "yes"
    assert device_ledger.read_ledger(device.serial) == []


def test_repeated_writes_keep_the_original_restore_point(tmp_path: Path) -> None:
    device = _device({"backend_env": "production"})
    engine = _engine(tmp_path, device)

    engine.prefs_write(PKG, PREFS_FILE, {"backend_env": "staging"}, relaunch=False)
    engine.prefs_write(PKG, PREFS_FILE, {"backend_env": "local"}, relaunch=False)
    assert device.prefs[PREFS_FILE]["backend_env"] == "local"

    outcome = device_ledger.replay(
        device.serial,
        context=device_ledger.UndoContext(
            serial=device.serial,
            device=device,
            capability=lambda name: flags_mod if name == FEATURE_FLAGS else None,
        ),
    )

    assert not outcome["failed"], outcome
    assert device.prefs[PREFS_FILE]["backend_env"] == "production"


def test_undoing_a_file_the_app_never_had_removes_it_rather_than_blanking_it(
    tmp_path: Path,
) -> None:
    """An empty `<map/>` is not the same state as "this app has never written this file".

    Some apps seed their defaults only on that first miss, so leaving an empty file behind is a
    different app than the one AUA was handed.
    """
    device = _device()
    engine = _engine(tmp_path, device)

    engine.prefs_write(PKG, PREFS_FILE, {"onboarding_seen": True}, relaunch=False)
    device_ledger.replay(
        device.serial,
        context=device_ledger.UndoContext(
            serial=device.serial,
            device=device,
            capability=lambda name: flags_mod if name == FEATURE_FLAGS else None,
        ),
    )

    assert PREFS_FILE not in device.prefs
    assert f"shared_prefs/{PREFS_FILE}" not in device.app_files


def test_the_mutation_is_registered_in_the_catalogue_with_a_real_undo() -> None:
    """The architecture guard's own question, asked about this feature specifically."""
    mutation = device_ledger.MUTATION_CATALOGUE["app_prefs"]

    assert mutation.undo_op == "restore_app_prefs"
    assert mutation.undo_op in device_ledger.UNDO_OPS
    assert device_ledger.catalogue_gaps() == []


# --------------------------------------------------------------------------- platform boundary


def test_the_capability_contract_names_the_write() -> None:
    assert {"snapshot_prefs", "save_prefs_backup", "write_prefs", "restore_prefs"} <= (
        CAPABILITY_METHODS[FEATURE_FLAGS]
    )


def test_the_android_adapter_still_satisfies_its_own_capability(tmp_path: Path) -> None:
    service = AndroidPlatform(make_config(cache={"dir": str(tmp_path)})).capability("feature_flags")

    assert service is flags_mod
    assert callable(service.write_prefs)


@register_platform("prefs-write-probe")
class _ProbePlatform(PlatformAdapter):
    """A non-Android adapter: the engine method must reach it and nothing else."""

    capabilities = frozenset({"feature_flags"})

    def connect(self, target_id: str | None = None) -> Device:
        raise AssertionError("the engine already holds a runtime")

    def list_targets(self) -> list[DeviceInfo]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree([])

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[dict[str, Any]] = []
        probe = self

        class _Service:
            build_uri = dump_result = load_flags_file = parse_assignments = staticmethod(
                lambda *a, **k: None
            )
            read_context_flags = read_prefs = staticmethod(lambda *a, **k: None)
            restore_prefs = staticmethod(lambda *a, **k: "restored")

            @staticmethod
            def snapshot_prefs(device: Device, package: str, file: str) -> Any:
                probe.calls.append({"call": "snapshot_prefs", "package": package, "file": file})
                return flags_mod.PrefsSnapshot(
                    package=package, file=f"{file}.xml", existed=False, xml=None
                )

            @staticmethod
            def save_prefs_backup(cache_dir: Any, serial: str, snapshot: Any) -> Path:
                probe.calls.append({"call": "save_prefs_backup", "serial": serial})
                path = Path(cache_dir) / "probe-backup.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                return path

            @staticmethod
            def write_prefs(
                device: Device,
                snapshot: Any,
                values: Any,
                *,
                relaunch: bool = True,
            ) -> dict[str, Any]:
                probe.calls.append(
                    {
                        "call": "write_prefs",
                        "package": snapshot.package,
                        "file": snapshot.file,
                        "values": dict(values),
                        "relaunch": relaunch,
                    }
                )
                return {"ok": True, "action": "prefs-write", "relaunched": relaunch}

        self._service = _Service()

    def load_capability(self, capability: str) -> object | None:
        return self._service if capability == "feature_flags" else None


def test_the_core_writes_prefs_through_whatever_platform_is_selected(tmp_path: Path) -> None:
    """No `if platform == "android"`: the engine only knows the capability."""
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, daemon={"enabled": False})
    platform = _ProbePlatform(cfg)
    engine = Engine(cfg, device=_device(), platform=platform)

    result = engine.prefs_write(PKG, "example_settings", {"a": "b"}, relaunch=False)

    assert result["ok"] is True
    assert [call["call"] for call in platform.calls] == [
        "snapshot_prefs",
        "save_prefs_backup",
        "write_prefs",
    ], "snapshot, journal, then write — the undo must exist before the mutation"
    assert platform.calls[-1] == {
        "call": "write_prefs",
        "package": PKG,
        "file": "example_settings.xml",
        "values": {"a": "b"},
        "relaunch": False,
    }
