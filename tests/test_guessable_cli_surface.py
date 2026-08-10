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


def test_screen_names_analyze_instead_of_running_it() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, ["screen"])
    assert result.exit_code != 0, "a synonym that works keeps the wrong name alive"
    combined = result.output + str(result.stderr or "")
    assert "analyze" in combined, f"the error must name the real command: {combined!r}"


def test_screen_does_not_render_a_plausible_help_page() -> None:
    """Same trap as the removed aliases: `--help` exiting 0 reads as "this command exists"."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, ["screen", "--help"])
    assert result.exit_code != 0
    assert "analyze" in result.output + str(result.stderr or "")


def test_input_text_is_answered_not_just_rejected() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        app, ["input-and-analyze", "--rid", "promptField", "--text", "hello"]
    )
    assert result.exit_code != 0, "--text still must not type; it selects by label elsewhere"
    combined = result.output + str(result.stderr or "")
    assert "positional" in combined, f"the error must say where the text goes: {combined!r}"
    assert "--index" not in combined, "click's nearest-name hint is the thing being replaced"


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
