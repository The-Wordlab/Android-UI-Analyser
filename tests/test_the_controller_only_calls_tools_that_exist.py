"""A tool rename must fail here, not on a leased device twenty minutes into a run.

`flags_apply` became `flags_apply_and_analyze`. The controller kept the old name, and the usage
error it got back arrived as "feature flags not applied and verified" - a precondition failure
dressed as a product one, discovered only after a full install, a guest sign-in and a recording.
"""

from __future__ import annotations

import re
from pathlib import Path

from android_ui_analyser.mcp_server import _tool_definitions

_CALL = re.compile(r'\bcall\(\s*"([a-z0-9_]+)"')
_CONTROLLER = Path(__file__).resolve().parents[1] / "experiments" / "aua_controller"


def test_every_tool_the_controller_names_is_a_tool_aua_publishes() -> None:
    published = {tool.name for tool in _tool_definitions()}
    called: dict[str, list[str]] = {}
    for path in sorted(_CONTROLLER.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for name in _CALL.findall(line):
                called.setdefault(name, []).append(f"{path.name}:{lineno}")

    assert called, "no tool calls found; the pattern stopped matching the controller"
    unknown = {name: where for name, where in called.items() if name not in published}
    assert not unknown, "the controller calls tools AUA does not publish:\n" + "\n".join(
        f"  {name} at {', '.join(where)}" for name, where in sorted(unknown.items())
    )
