"""What an agent reaches for before it reads the guide, and what happens when it does.

Every case here is a real wrong guess made by an agent driving a live app, not a hypothetical.
The pattern was the same each time: one command worked on the first try, which read as "I know
this tool", and the rest was guessed from the shape of the previous line. A wrong guess is cheap
to answer well and expensive to answer badly — click's nearest-name hint sent `screen` to
`screenshot` (a PNG, not the element list) and `--text` to `--index`, both further from the fix.

- ``screen`` is what gets typed instead of ``analyze``. It is registered so it can FAIL naming
  ``analyze`` — not aliased. An alias that quietly works leaves the wrong name in agent memory,
  which is the same reasoning that removed the short action aliases; the only thing being fixed
  here is Click's hint, which sends ``screen`` to ``screenshot`` (a PNG, not the element list).
- ``input-and-analyze --text "foo"`` is assumed from ``--rid``/``--desc`` being flags; the text is
  positional, and ``--text`` cannot be reused because it selects by label on every other command.
  It is accepted purely so the error can say that.
- ``--format tsv`` was honoured by ``analyze`` and silently ignored by every action, so an agent
  that had settled on TSV got JSON back the moment it tapped anything.
"""

from __future__ import annotations

from android_ui_analyser.cli import app
from android_ui_analyser.projection import Projection, render_action_tsv
from android_ui_analyser.schema import OutputFormat


def _invoke(*args: str) -> tuple[int, dict]:
    import json

    from typer.testing import CliRunner

    result = CliRunner().invoke(app, list(args))
    combined = result.output + str(result.stderr or "")
    payload: dict = {}
    for line in combined.splitlines():
        if line.startswith("{"):
            payload = json.loads(line).get("error", {})
            break
    return result.exit_code, payload


def test_a_guessed_name_is_answered_with_the_one_that_was_meant() -> None:
    from android_ui_analyser.guide import COMMAND_SYNONYMS

    for guess, meant in COMMAND_SYNONYMS.items():
        code, err = _invoke(guess)
        assert code != 0, f"`aua {guess}` must fail, not quietly work"
        assert err.get("code") == "unknown_command", f"{guess}: {err!r}"
        assert err.get("did_you_mean") == meant, f"{guess} should point at {meant}: {err!r}"


def test_every_synonym_points_at_a_command_that_exists() -> None:
    """A wrong name answered with another wrong name is worse than Click's guess."""
    from typer.main import get_command

    from android_ui_analyser.guide import COMMAND_SYNONYMS

    real = set(get_command(app).commands)
    unknown = {m for m in COMMAND_SYNONYMS.values() if m not in real}
    assert not unknown, f"synonyms point at non-existent commands: {unknown}"


def test_no_synonym_shadows_a_real_command() -> None:
    from typer.main import get_command

    from android_ui_analyser.guide import COMMAND_SYNONYMS

    real = set(get_command(app).commands)
    shadowed = {g for g in COMMAND_SYNONYMS if g in real}
    assert not shadowed, f"these are real commands and would never reach the handler: {shadowed}"


def test_an_unrecognised_name_still_gets_the_orientation() -> None:
    code, err = _invoke("zzzznope")
    assert code != 0
    assert "did_you_mean" not in err, "do not invent a mapping we do not have"
    how = err.get("how_to_drive") or []
    assert any("analyze" in line for line in how), f"an empty answer is the failure mode: {how!r}"
    assert any("--until" in line for line in how), "the sleep habit is what this is for"


def test_input_text_is_answered_not_just_rejected() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        app, ["input-and-analyze", "--rid", "promptField", "--text", "hello"]
    )
    assert result.exit_code != 0, "--text still must not type; it selects by label elsewhere"
    combined = result.output + str(result.stderr or "")
    assert "positional" in combined, f"the error must say where the text goes: {combined!r}"
    assert "--index" not in combined, "click's nearest-name hint is the thing being replaced"
    # The example must not teach a name that now exits 2 as a removed alias.
    assert "`aua input-and-analyze --rid" in combined, f"stale example: {combined!r}"


def test_a_lone_positional_is_told_it_was_eaten_as_the_target() -> None:
    """"input needs the text to type" reads as false to a caller who passed text."""
    code, err = _invoke("input-and-analyze", "zzzqqqxyz")
    assert code != 0
    assert "zzzqqqxyz" in err.get("message", ""), f"echo their argument back: {err!r}"
    assert "element" in err.get("message", ""), f"name what it was read as: {err!r}"
    hint = err.get("hint", "")
    assert "--rid" in hint and "zzzqqqxyz" in hint, f"the fix must be copy-pasteable: {hint!r}"


def test_fields_on_an_action_becomes_observe_fields() -> None:
    """Learned on `analyze`, typed on `tap-and-analyze`; it is the same projection."""
    from android_ui_analyser.cli import alias_fields_on_actions

    assert alias_fields_on_actions(["tap-and-analyze", "--rid", "x", "--fields", "id,text"]) == [
        "tap-and-analyze",
        "--rid",
        "x",
        "--observe-fields",
        "id,text",
    ]
    assert alias_fields_on_actions(["tap-and-analyze", "--fields=id,text"]) == [
        "tap-and-analyze",
        "--observe-fields=id,text",
    ]


def test_analyze_keeps_its_own_fields() -> None:
    from android_ui_analyser.cli import alias_fields_on_actions

    argv = ["analyze", "--fields", "id,text"]
    assert alias_fields_on_actions(argv) == argv, "the command that defines it must win"


def test_leading_globals_do_not_hide_the_subcommand() -> None:
    """`_first_subcommand` exists because a bare scan lands on a global's value, not the command."""
    from android_ui_analyser.cli import alias_fields_on_actions

    argv = ["--serial", "emulator-5554", "analyze", "--fields", "id,text"]
    assert alias_fields_on_actions(argv) == argv, "analyze is still the target here"

    hoisted = alias_fields_on_actions(["--format", "tsv", "tap-and-analyze", "--fields", "id"])
    assert "--observe-fields" in hoisted


def _action_payload() -> dict:
    return {
        "ok": True,
        "action": "tap",
        "id": 40,
        "detail": None,
        "change": {
            "activity_changed": False,
            "text_added": ["Create your account", "Continue with Google"],
            "text_removed": ["No apps found"],
        },
        "next_actions": [{"id": 2, "label": "Continue with Google"}],
        "observation": {
            "schema_version": 1,
            "screen": {"width": 1080, "height": 2400, "package": "com.example.app"},
            "elements": [
                {"id": 2, "text": "Continue with Google", "clickable": True, "window": "app"},
                {"id": 3, "text": "Maybe later", "clickable": True, "window": "app"},
            ],
            "meta": {"tier_used": "hierarchy", "duration_ms": 99},
        },
    }


def test_an_action_renders_its_elements_as_tsv_rows() -> None:
    lines = render_action_tsv(_action_payload()).splitlines()
    rows = [ln for ln in lines if not ln.startswith("#")]
    assert rows, "the whole point of tsv is one element per line"
    assert rows[0].split("\t")[:2] == ["id", "text"], f"expected a header row, got {rows[0]!r}"
    assert any("Continue with Google" in row for row in rows[1:])


def test_the_action_verdict_survives_as_comments() -> None:
    lines = render_action_tsv(_action_payload()).splitlines()
    comments = [ln for ln in lines if ln.startswith("#")]
    assert "# action=tap" in comments
    assert "# ok=true" in comments
    # `change.text_added` is usually the entire reason the action was run — it is the cheapest
    # possible assertion that a tap did what it was supposed to do.
    assert any(
        ln.startswith("# change.text_added=") and "Create your account" in ln for ln in comments
    ), f"the verdict must stay greppable, got {comments!r}"


def test_a_null_envelope_field_is_not_rendered() -> None:
    lines = render_action_tsv(_action_payload()).splitlines()
    assert not any(ln.startswith("# detail=") for ln in lines), "empty is reserved for unknown"


def test_the_observation_view_is_honoured_when_one_was_asked_for() -> None:
    view = Projection.for_observation("id,text", fmt=OutputFormat.tsv)
    lines = render_action_tsv(_action_payload(), view).splitlines()
    header = next(ln for ln in lines if not ln.startswith("#"))
    assert header.split("\t") == ["id", "text"], f"--observe-fields must win, got {header!r}"


def test_an_action_without_an_observation_still_renders_its_envelope() -> None:
    out = render_action_tsv({"ok": True, "action": "key", "detail": "BACK"})
    assert "# action=key" in out.splitlines()
