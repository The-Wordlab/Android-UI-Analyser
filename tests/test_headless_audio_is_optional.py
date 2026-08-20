"""A headless start silenced the device unconditionally, so no scenario could ask for sound.

`-no-audio` was appended to every headless start, which is every start the pool makes. A
scenario whose subject is audio therefore had no audio device to observe, and no flag to ask
for one.

Two corrections to how this was written up. It was never passed on *every* start - only
headless ones - though since the pool is always headless the effect was the same. And it did
not make audio unverifiable: `dumpsys audio` reports an AudioPlaybackConfiguration reaching
state:started within ~30ms of a play tap, and `dumpsys media.audio_flinger`'s signal power
history separates real output from an idle stream, which is how the sweep's clearest
FAIL_CRITICAL was actually measured. Only microphone input, and so transcript content, is
genuinely unobservable.

Silent stays the default - one less subsystem on a machine running five workers - so these
tests pin both directions, the default and the opt-in.
"""

from __future__ import annotations

import inspect

from android_ui_analyser import emulator


def _headless_branch() -> str:
    """The source of `start`, which is where the argv is assembled."""
    return inspect.getsource(emulator.start)


def test_audio_defaults_to_off():
    """The pool's behaviour must not change just because the flag now exists."""
    assert inspect.signature(emulator.start).parameters["audio"].default is False


def test_no_audio_is_no_longer_unconditional():
    src = _headless_branch()
    # The silencing flag must be reached through the parameter, not appended outright.
    assert '"-no-audio"' in src
    assert 'if not audio:' in src
    unconditional = '"-no-window", "-no-audio"'
    assert unconditional not in src, "-no-audio is still glued to the headless argv"


def test_the_other_headless_flags_are_still_unconditional():
    """`-no-window` and `-no-boot-anim` are not what this change is about."""
    src = _headless_branch()
    assert '"-no-window"' in src
    assert '"-no-boot-anim"' in src


def test_the_cli_exposes_the_choice():
    from android_ui_analyser.cli import emulator_start_cmd

    params = inspect.signature(emulator_start_cmd).parameters
    assert "audio" in params, "`emulator start` must be able to ask for an audio device"
    assert params["audio"].default.default is False
