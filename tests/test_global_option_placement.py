"""A global option written after the subcommand must still work.

`aua analyze --format json` is the single most repeated mistake in this project - across
sessions, models, and a warning documented on day one. Click is right that `--format` binds
to the group, but being right costs a failed call, a wasted agent turn, and a detour into
`--help`, on an invocation whose intent was never ambiguous.

The care is in what must NOT move: several commands reuse a global name for their own
purpose (`emulator stop --serial` names the emulator to kill; an export command's `--format`
means gif|mp4). Silently hoisting those would turn a working command into a subtly wrong
one, which is far worse than the error message this replaces.
"""

from __future__ import annotations

from android_ui_analyser.cli import hoist_global_options as hoist


def test_format_after_subcommand_is_hoisted():
    assert hoist(["analyze", "--format", "json"]) == ["--format", "json", "analyze"]


def test_equals_form_is_hoisted():
    assert hoist(["analyze", "--format=json"]) == ["--format=json", "analyze"]


def test_already_correct_order_is_untouched():
    argv = ["--format", "json", "analyze"]
    assert hoist(argv) == argv


def test_subcommand_options_stay_put():
    assert hoist(["analyze", "--source", "vision", "--format", "json"]) == [
        "--format", "json", "analyze", "--source", "vision",
    ]


def test_analyze_keeps_its_own_no_cache():
    """`analyze` declares --no-cache itself, so it must not be treated as the global one.

    They mean nearly the same thing here, which is exactly why moving it would be hard to
    notice - and the whole point of the collision check is that "nearly" is not "exactly".
    """
    argv = ["analyze", "--no-cache"]
    assert hoist(argv) == argv


def test_boolean_global_is_hoisted_without_eating_the_next_token():
    """A value-less global must not swallow the following token."""
    assert hoist(["devices", "--no-cache"]) == ["--no-cache", "devices"]


def test_serial_is_hoisted_for_a_command_that_does_not_define_it():
    assert hoist(["analyze", "--serial", "emulator-5554"]) == [
        "--serial", "emulator-5554", "analyze",
    ]


# ---- the cases that must NOT move ----


def test_emulator_stop_keeps_its_own_serial():
    """`emulator stop --serial` names the emulator to kill, not the device to talk to."""
    argv = ["emulator", "stop", "--serial", "emulator-5554"]
    assert hoist(argv) == argv


def test_dashboard_keeps_its_own_serial():
    argv = ["dashboard", "--serial", "emulator-5554"]
    assert hoist(argv) == argv


def test_nothing_after_a_double_dash_is_touched():
    argv = ["shell-ish", "--", "--format", "json"]
    assert hoist(argv)[-3:] == ["--", "--format", "json"]


def test_bare_global_flags_with_no_subcommand_are_untouched():
    argv = ["--version"]
    assert hoist(argv) == argv


def test_empty_argv():
    assert hoist([]) == []


def test_unknown_option_is_left_for_click_to_report():
    """Not our job to guess; a real typo must still produce Click's error."""
    argv = ["analyze", "--totally-made-up"]
    assert hoist(argv) == argv


# --------------------------------------------------------------------------- value skipping


def test_a_preceding_global_value_is_not_mistaken_for_the_subcommand():
    """`--serial emulator-5558 analyze --format json` must hoist just like the bare form.

    The scan for "first token that is not an option" used to land on `emulator-5558` — the
    *value* of `--serial`, at index 1 — so every later option looked like it already sat
    before the subcommand and nothing moved. That is the shape agents always write, because
    they pass `--serial` on every call, so the hoist quietly did nothing in exactly the case
    it was built for.
    """
    assert hoist(["--serial", "emulator-5558", "analyze", "--format", "json"]) == [
        "--format", "json", "--serial", "emulator-5558", "analyze",
    ]


def test_inline_global_value_still_finds_the_subcommand():
    assert hoist(["--serial=emulator-5558", "analyze", "--format", "json"]) == [
        "--format", "json", "--serial=emulator-5558", "analyze",
    ]


def test_valueless_global_before_the_subcommand():
    assert hoist(["--no-cache", "tap", "9", "--format", "compact"]) == [
        "--format", "compact", "--no-cache", "tap", "9",
    ]


# --------------------------------------------------------------------------- action globals


def test_until_after_the_action_is_hoisted():
    """`tap --rid x --until "text:Chats"` is how anyone writes it.

    An agent wrote exactly that, got "No such option '--until'", detoured into `tap --help`,
    and fell back to the `--no-observe` + `analyze` pair these flags exist to replace.
    """
    assert hoist(["tap", "--rid", "x", "--until", "text:Chats"]) == [
        "--until", "text:Chats", "tap", "--rid", "x",
    ]


def test_until_with_serial_and_format_around_it():
    assert hoist(
        ["--serial", "emulator-5558", "tap", "--rid", "x", "--until", "text:Chats"]
    ) == ["--until", "text:Chats", "--serial", "emulator-5558", "tap", "--rid", "x"]


def test_until_timeout_and_poll_carry_their_values():
    assert hoist(["tap", "9", "--until", "rid:card", "--until-timeout", "45000"]) == [
        "--until", "rid:card", "--until-timeout", "45000", "tap", "9",
    ]


def test_observe_fields_after_the_action_is_hoisted():
    assert hoist(["key", "back", "--observe-fields", "all"]) == [
        "--observe-fields", "all", "key", "back",
    ]
