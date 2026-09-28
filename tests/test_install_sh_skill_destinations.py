"""Current Codex discovery stays available without moving legacy client configuration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_codex_skill_install_destinations_preserve_custom_legacy_home(tmp_path):
    installer = Path(__file__).resolve().parents[1] / "install.sh"
    legacy = tmp_path / "custom-codex"
    result = subprocess.run(
        ["bash", str(installer), "--print-plan"],
        env={**os.environ, "CODEX_HOME": str(legacy)},
        text=True,
        capture_output=True,
        check=True,
    )
    plan = {}
    for row in result.stdout.splitlines():
        name, separator, value = row.partition(":")
        if separator:
            plan[name.strip()] = value.strip()
    assert plan["codex-skill"] == str(Path.home() / ".agents/skills/android-ui-analyser")
    assert plan["codex-legacy-skill"] == str(legacy / "skills/android-ui-analyser")
    assert not legacy.exists(), "print-plan must not create an installation"
