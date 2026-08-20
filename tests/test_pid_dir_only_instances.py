"""The pid directory holds more than instance records, so readers must filter.

`_pid_dir` is shared: `anim_off` writes a settings snapshot next to the instance records as
`<inst>.anim.json` so the device's animation and dialog settings can be restored later. Two readers -
`emulator status` and `_aua_started_records` - globbed `*.json` and treated every file as a started
emulator.

Found on a machine that had been running the QA pool for two days: `aua emulator status` reported
**24 "started by aua" emulators, every one a settings snapshot**, while no emulator was running at
all. A coordinator deciding which AVD was free could not parse a single entry, so its own check
concluded the registry was empty and declared the whole pool safe to dispatch to — a false "free",
which is the expensive direction.

The snapshots also never get cleaned up, because the stop and cleanup paths match on `serial`, `pid`
or `avd` and a snapshot carries none of those. That leak is what let 24 accumulate and made this
visible; these tests pin the filtering, not the leak.
"""

from __future__ import annotations

import json

from android_ui_analyser import emulator

INSTANCE = {
    "avd": "aua_qa_1",
    "instance": "aua_qa_1.p5554",
    "port": 5554,
    "serial": "emulator-5554",
    "started_by_aua": True,
    "started_at": 1.0,
    "last_activity": 2.0,
}

# Exactly what devopts.anim_off writes, and what was found polluting the directory.
SNAPSHOT = {
    "anim": {
        "window_animation_scale": "0",
        "transition_animation_scale": "0",
        "animator_duration_scale": "0",
    },
    "crashes_visible": True,
    "hide_error_dialogs": "0",
    "anr_show_background": "0",
    "dont_keep_activities": False,
    "always_finish_activities": "0",
}


def _seed(cache_dir, *, instances=1, snapshots=3):
    d = emulator._pid_dir(cache_dir)
    for i in range(instances):
        meta = dict(INSTANCE, avd=f"aua_qa_{i + 1}", serial=f"emulator-{5554 + 2 * i}")
        (d / f"aua_qa_{i + 1}.p{5554 + 2 * i}.json").write_text(json.dumps(meta))
    for i in range(snapshots):
        (d / f"aua_qa_{i + 1}.p{5554 + 2 * i}.anim.json").write_text(json.dumps(SNAPSHOT))
    return d


def test_a_settings_snapshot_is_not_an_instance_record():
    assert emulator._is_instance_record(INSTANCE) is True
    assert emulator._is_instance_record(SNAPSHOT) is False


def test_non_dict_and_empty_avd_are_not_instances():
    """A record whose `avd` is missing or blank cannot identify a device, so it is not one."""
    for junk in (None, [], "aua_qa_1", 5554, {}, {"avd": ""}, {"avd": None}, {"serial": "x"}):
        assert emulator._is_instance_record(junk) is False


def test_started_records_ignores_snapshots(tmp_path):
    _seed(tmp_path, instances=2, snapshots=3)
    got = emulator._aua_started_records(tmp_path)
    assert len(got) == 2, f"expected 2 instances, got {len(got)}: {got}"
    assert {r["avd"] for r in got} == {"aua_qa_1", "aua_qa_2"}


def test_started_records_is_empty_when_only_snapshots_exist(tmp_path):
    """The case that was found in the wild: no emulators running, 24 snapshots on disk."""
    _seed(tmp_path, instances=0, snapshots=24)
    assert emulator._aua_started_records(tmp_path) == []


def test_a_snapshot_alongside_an_instance_does_not_hide_it(tmp_path):
    """The dangerous shape: one real instance, unparseable neighbours.

    A reader that gives up on the directory would report nothing held and a caller would treat a
    busy AVD as free.
    """
    _seed(tmp_path, instances=1, snapshots=5)
    got = emulator._aua_started_records(tmp_path)
    assert [r["avd"] for r in got] == ["aua_qa_1"]
    assert all("anim" not in r for r in got)
