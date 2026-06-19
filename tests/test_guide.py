"""AC15 — ``aua guide``: the agent manual covers the session protocol, escalation ladder,
memory, schema, and exit codes; ``--json`` / ``--brief`` work; ``aua --help`` references it;
and the emitted ``SKILL.md`` is produced from the same source (no drift).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from android_ui_analyser import guide
from android_ui_analyser.cli import app

runner = CliRunner()


# --------------------------------------------------------------------------- content


def test_markdown_covers_required_topics() -> None:
    low = guide.render_markdown().lower()
    for needle in [
        "session protocol",
        "daemon",
        "aua map",
        "wait --for-stable",
        "analyze",
        "tap",
        "escalation",
        "memory",
        "known_screen",
        "schema",
        "exit codes",
    ]:
        assert needle in low, needle


def test_brief_is_shorter_but_keeps_the_protocol() -> None:
    full = guide.render_markdown()
    brief = guide.render_brief()
    assert "Session protocol" in brief
    assert len(brief) < len(full)


def test_json_is_structured() -> None:
    j = guide.render_json()
    assert {"session_protocol", "escalation_ladder", "exit_codes", "schema_fields"} <= set(j)
    assert any("daemon" in step["detail"].lower() for step in j["session_protocol"])
    assert any("grounding" in row["tier"].lower() for row in j["escalation_ladder"])


# --------------------------------------------------------------------------- no drift


def test_emit_skill_matches_render_skill_no_drift(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "SKILL.md"
    written = guide.emit_skill(out)
    assert written == out and out.is_file()
    text = out.read_text()
    assert text == guide.render_skill()
    # The skill BODY is exactly the rendered guide manual — single source, no drift.
    assert guide.render_markdown(brief=False) in text


def test_skill_frontmatter_has_name_and_trigger_description() -> None:
    skill = guide.render_skill()
    assert skill.startswith("---")
    front = skill.split("---", 2)[1]
    import yaml

    meta = yaml.safe_load(front)
    assert meta["name"] == "android-ui-analyser"
    assert "android" in meta["description"].lower()  # trigger description preserved


# --------------------------------------------------------------------------- CLI


def test_cli_help_references_guide() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "guide" in res.stdout.lower()


def test_cli_guide_default_brief_and_json() -> None:
    res = runner.invoke(app, ["guide"])
    assert res.exit_code == 0 and "Session protocol" in res.stdout

    brief = runner.invoke(app, ["guide", "--brief"])
    assert brief.exit_code == 0 and "Session protocol" in brief.stdout

    js = runner.invoke(app, ["--format", "json", "guide", "--json"])
    assert js.exit_code == 0
    assert "session_protocol" in json.loads(js.stdout)


def test_cli_emit_skill_explicit_path(tmp_path: Path) -> None:
    out = tmp_path / "SKILL.md"
    res = runner.invoke(app, ["guide", "--emit-skill", str(out)])
    assert res.exit_code == 0, res.stderr
    assert out.read_text() == guide.render_skill()


def test_cli_emit_skill_bare_uses_default_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["guide", "--emit-skill"])
    assert res.exit_code == 0, res.stderr
    out = tmp_path / ".claude" / "skills" / "android-ui-analyser" / "SKILL.md"
    assert out.is_file()
    assert out.read_text() == guide.render_skill()
