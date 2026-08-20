"""`doctor` must notice when the *installed* skill no longer matches `guide.py`.

The pre-commit hook keeps the two in-repo SKILL.md copies in sync, but nothing syncs the
copy agents actually load — `~/.claude/skills/…` only changes when someone re-runs
`install.sh`. Observed: a guide change was committed, both repo copies updated, and live
agents kept reading a day-old skill for hours. Nothing anywhere reported it.

The failure is silent by construction: an agent following stale instructions does not error,
it reaches for flags that no longer exist and misses the ones that do.
"""

from __future__ import annotations

from pathlib import Path

import android_ui_analyser.cli as cli
from android_ui_analyser import guide as guide_mod


def _point_at(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(cli, "_USER_SKILL", path)


def test_matching_skill_passes(monkeypatch, tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(guide_mod.render_skill(), encoding="utf-8")
    _point_at(monkeypatch, skill)
    assert cli._installed_skill_check()["ok"] is True


def test_drifted_skill_fails_and_says_how_to_fix_it(monkeypatch, tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(guide_mod.render_skill() + "\n<!-- drift -->\n", encoding="utf-8")
    _point_at(monkeypatch, skill)
    out = cli._installed_skill_check()
    assert out["ok"] is False
    assert "emit-skill" in out.get("hint", ""), "a failure must carry the one-line fix"


def test_absent_skill_is_not_a_failure(monkeypatch, tmp_path):
    """Not every install wants the user-level copy; absence is a choice, not breakage."""
    _point_at(monkeypatch, tmp_path / "nope" / "SKILL.md")
    assert cli._installed_skill_check()["ok"] is True


def test_doctor_checks_claude_and_codex_install_targets(monkeypatch, tmp_path):
    claude = tmp_path / "claude" / "SKILL.md"
    codex = tmp_path / "codex" / "SKILL.md"
    claude.parent.mkdir(parents=True)
    codex.parent.mkdir(parents=True)
    claude.write_text(guide_mod.render_skill(), encoding="utf-8")
    codex.write_text(guide_mod.render_skill() + "\nold", encoding="utf-8")
    monkeypatch.setattr(cli, "_CLAUDE_USER_SKILL", claude)
    monkeypatch.setattr(cli, "_CODEX_USER_SKILL", codex)

    checks = cli._installed_skill_checks()

    assert checks["claude"]["ok"] is True
    assert checks["codex"]["ok"] is False
    assert checks["ok"] is False


def test_check_is_wired_into_the_pretty_report():
    rendered = cli._render_doctor_pretty(
        {"checks": {"skill": {"ok": False, "detail": "stale", "hint": "aua guide --emit-skill X"}}}
    )
    assert "skill" in rendered and "stale" in rendered
    assert "aua guide --emit-skill X" in rendered, "the fix must be visible, not just the fault"
