"""An observation handed to a caller must leave its ids resolvable.

Numeric element ids resolve through one cache file per device, shared by every caller of
that device. So "publish numbered elements" and "record those numbers" are the same act; a
surface that does the first without the second hands out ids that the next action validates
against somebody else's screen. The dashboard did exactly that — deliberately, to avoid
overwriting an agent's ids — and every click it sent came back as a changed binding naming
an element the clicker had never seen.

``record_ids=False`` exists for the opposite case: AUA's own freshness reads, whose
observation is never returned to anyone. It therefore belongs to the engine alone, and no
caller-facing surface may suppress recording.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "android_ui_analyser"

# Surfaces that hand observations to a caller: a human at a terminal, an agent over MCP,
# a reader in a browser, or another process through the daemon.
CALLER_FACING = ("cli.py", "mcp_server.py", "dashboard.py", "daemon.py")


def test_only_the_engine_may_suppress_id_recording() -> None:
    offenders = [
        name
        for name in CALLER_FACING
        if "record_ids=False" in (SOURCE / name).read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "these publish observations to a caller, so their ids must be recorded: "
        + ", ".join(offenders)
    )


def test_the_dashboard_never_asks_for_an_unrecorded_analyze() -> None:
    """``no_cache`` used to mean both "do not reuse" and "do not record"; only the first now."""
    text = (SOURCE / "dashboard.py").read_text(encoding="utf-8")
    calls = re.findall(r'"analyze",\n(?:\s+.*\n)*?\s+\)', text)
    assert calls, "the dashboard no longer runs an analyze; this guard would pass forever"
    assert not [call for call in calls if "no_cache" in call], (
        "a dashboard analyze draws numbered boxes for a human to click, so its ids have to "
        "be recorded — withholding them is what made every click unresolvable"
    )


def test_the_engine_still_documents_why_it_opts_out() -> None:
    """The exemption is narrow on purpose, so it must stay explained where it is used."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(SOURCE.glob("engine*.py")))
    assert "record_ids=False" in text, "no internal read opts out; has the seam been removed?"
    assert "record_ids:" in text, "the engine no longer accepts the parameter it threads"
