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
        "agent best practices",
        "clipboard paste",
        "from-here",
        "hierarchy-only",
        "aua db",
        "restore point",
    ]:
        assert needle in low, needle


def test_guide_teaches_agents_to_choose_verified_navigation_first() -> None:
    manual = guide.render_markdown()

    observe = manual.index("Observe once, then choose the highest-level action")
    playbook = manual.index("Read the app playbook when the task needs it")
    deeplinks = manual.index("Take shortcuts with deeplinks")
    assert observe < playbook < deeplinks

    decision = dict(guide.SESSION_PROTOCOL)["Observe once, then choose the highest-level action"]
    assert decision.index("# goto:") < decision.index("# flows:")
    assert decision.index("aua goto") < decision.index("aua flow run")
    assert decision.index("aua flow run") < decision.index("aua open-and-analyze")
    assert decision.index("aua open-and-analyze") < decision.index("manual element actions")

    deeplink_detail = dict(guide.SESSION_PROTOCOL)["Take shortcuts with deeplinks"]
    assert "delivered intent is not proof of arrival" in deeplink_detail
    assert "stale_risk" in deeplink_detail

    orientation_commands = [command for command, _detail in guide.ORIENTATION]
    goto = next(
        i for i, command in enumerate(orientation_commands) if command.startswith("aua goto")
    )
    flow = next(
        i for i, command in enumerate(orientation_commands) if command.startswith("aua flow run")
    )
    tap = next(
        i
        for i, command in enumerate(orientation_commands)
        if command.startswith("aua tap-and-analyze")
    )
    assert goto < flow < tap


def test_guide_explains_that_automatic_leases_follow_the_agent_process() -> None:
    detail = dict(guide.SESSION_PROTOCOL)["One agent, one emulator — leases are automatic"]
    for claim in [
        "no owner prompt",
        "PID plus start token",
        "dead owner",
        "immediately treated as free",
    ]:
        assert claim in detail


def test_brief_is_shorter_but_keeps_the_protocol() -> None:
    full = guide.render_markdown()
    brief = guide.render_brief()
    assert "Session protocol" in brief
    assert len(brief) < len(full)
    assert len(brief.split()) < 1_200, "brief must be a decision loop, not the full manual"
    assert len(brief) < len(full) * 0.45


def test_brief_prefers_stable_selectors_and_current_playbook_facts() -> None:
    brief = guide.render_brief()
    for claim in [
        "stable_key",
        "prefer `--rid",
        "numeric id after its frame changed",
        "knowledge stale",
        "near-duplicate",
        "map --audit --summary",
        "text:Hello\\, friend",
        "daemon_outcome_unknown",
        "competing controller",
    ]:
        assert claim in brief


def test_json_is_structured() -> None:
    j = guide.render_json()
    assert {"session_protocol", "escalation_ladder", "exit_codes", "schema_fields"} <= set(j)
    assert any("daemon" in step["detail"].lower() for step in j["session_protocol"])
    assert any("grounding" in row["tier"].lower() for row in j["escalation_ladder"])


def test_mcp_initialization_teaches_uncertain_daemon_outcomes() -> None:
    from android_ui_analyser.capabilities import render_mcp_instructions

    text = render_mcp_instructions()
    assert "daemon_outcome_unknown" in text
    assert "never repeat" in text


# --------------------------------------------------------------------------- no drift


def test_emit_skill_matches_render_skill_no_drift(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "SKILL.md"
    written = guide.emit_skill(out)
    assert written == out and out.is_file()
    text = out.read_text()
    assert text == guide.render_skill()
    assert guide.render_skill_markdown() in text
    assert guide.render_markdown(brief=False) not in text


def test_generated_skill_is_brief_and_progressively_reveals_the_manual() -> None:
    skill = guide.render_skill()
    full = guide.render_markdown(brief=False)

    assert len(skill.encode("utf-8")) < 5 * 1024
    assert len(full.encode("utf-8")) > 50 * 1024
    assert "aua guide --brief" in skill
    assert "Run `aua guide`" in skill
    assert "aua capabilities --goal" in skill
    assert "restore point" in full, "full reference content must not be discarded"


def test_guide_defines_authoritative_call_accounting_without_inflating_calls() -> None:
    protocol = dict(guide.SESSION_PROTOCOL)["Start from the user's goal"]
    brief = guide.render_brief()
    skill = guide.render_skill_markdown()

    for text in (protocol, brief, skill):
        for field in [
            "`journal_events`",
            "`top_level_calls`",
            "`folded_internal_events`",
            "`lifecycle_calls`",
            "`task_calls`",
            "`reporting_call_included`",
            "`top_level_calls_including_reporting_call`",
        ]:
            assert field in text
        assert "caller-visible invocations" in text
        assert "action-bound wait" in text
        assert "reporting_call_included` is false" in text

    assert "authoritative instead of estimating" in protocol
    assert "Use `review.accounting`, not estimates" in skill


def test_flow_capture_proof_and_selector_resilience_are_taught_everywhere() -> None:
    manual = guide.render_markdown()
    skill = guide.render_skill_markdown()
    cli_help = runner.invoke(app, ["flow", "save", "--help"])

    assert cli_help.exit_code == 0
    assert "selector resilience" in cli_help.stdout
    for text in (manual, skill):
        assert "selector_resilience" in text
        assert "satisfied_action_until" in text
        assert "privacy-safe positive `--until`" in text


def test_session_review_help_names_exact_accounting_partitions() -> None:
    help_result = runner.invoke(app, ["session", "review", "--help"])

    assert help_result.exit_code == 0
    normalized = " ".join(help_result.stdout.split())
    assert "top-level/task/lifecycle calls" in normalized
    assert "folded events" in normalized


def test_both_committed_skill_copies_match_the_compact_generator() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = guide.render_skill()

    assert (root / ".claude/skills/android-ui-analyser/SKILL.md").read_text() == expected
    assert (root / "skills/android-ui-analyser/SKILL.md").read_text() == expected


def test_skill_frontmatter_has_name_and_trigger_description() -> None:
    skill = guide.render_skill()
    assert skill.startswith("---")
    front = skill.split("---", 2)[1]
    import yaml

    meta = yaml.safe_load(front)
    assert meta["name"] == "android-ui-analyser"
    assert "android" in meta["description"].lower()  # trigger description preserved


def test_codex_metadata_and_bundle_share_the_canonical_skill(tmp_path: Path) -> None:
    root = tmp_path / "android-ui-analyser"
    written = guide.emit_skill_bundle(root)

    assert written == root
    assert (root / "SKILL.md").read_text(encoding="utf-8") == guide.render_skill()
    metadata = (root / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert metadata == guide.render_codex_agent_metadata()
    assert "$android-ui-analyser" in metadata


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


def test_cli_emits_codex_metadata(tmp_path: Path) -> None:
    out = tmp_path / "agents" / "openai.yaml"
    res = runner.invoke(app, ["guide", "--emit-codex-metadata", str(out)])
    assert res.exit_code == 0, res.stderr
    assert out.read_text(encoding="utf-8") == guide.render_codex_agent_metadata()
