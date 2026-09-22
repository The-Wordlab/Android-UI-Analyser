"""A host action AUA refuses as stale is a non-event: no error, no step, no escalation.

The screen is read, the navigator picks a control, and by the time the press arrives the
app has moved on. AUA's stale-target guard sends nothing and answers ``element_not_found``
with a recovery observation. Seen live on the first row of a run: the navigator pressed
"log in" (1.0), the post-press read still showed the landing page because the login was in
flight, it pressed "log in" again (0.93), and the guard refused because the chat had
arrived meanwhile. That refusal was then counted as a tool error, fed to the chat model as
a failed host step, kept in the navigator's own journey as something it did, and it cost a
step of the budget -- for a press that never happened.

What should happen instead: mark the attempt ignored, forget it in the navigator's story,
read the screen again, and ask the same navigator about the screen as it is now.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_aua_controller_host_loop import TOOLS, action, observation, records, run  # noqa: E402

READ = {"type": "function", "function": {"name": "analyze_screen", "parameters": {
    "type": "object", "properties": {}, "additionalProperties": False}}}


def refusal(label="Between"):
    """AUA's pre-dispatch refusal, in the exact shape ``definitive_selector_miss`` recognises."""
    return {"ok": False, "error": {
        "code": "element_not_found",
        "message": "could not establish a unique current element for handle 'el:1'",
        "hint": "No action was sent. The element may be absent, indistinguishable from another item, changed, or from an expired target lifetime.",
        "observation_present": True, "observation": observation(label),
    }}


class Navigator:
    """Acts once, is asked again, remembers whether it was told to forget."""

    def __init__(self):
        self.seen: list[str] = []
        self.forgotten = 0

    async def __call__(self, latest):
        frame = latest.get("observation") or latest["error"]["observation"]
        self.seen.append(frame["meta"]["fingerprint"])
        return action("el:1") if len(self.seen) == 1 else None

    def forget(self):
        self.forgotten += 1


def test_a_refused_stale_press_is_ignored_reread_and_asked_again(tmp_path):
    navigator = Navigator()
    calls = []

    async def execute(tool, arguments):
        calls.append((tool, copy.deepcopy(arguments)))
        if tool == "tap":
            return refusal()
        return {"ok": True, "observation": observation("Fresh")}

    report, requests, _ = run(tmp_path, navigator, execute, tools=TOOLS + [READ], max_steps=1)

    # The press was refused, the screen was read again, and the same navigator was asked about
    # the fresh screen -- not the mid-transition frame the refusal carried.
    assert calls == [("tap", {"id": "el:1"}), ("analyze_screen", {})]
    assert navigator.seen == ["Start", "Fresh"]
    assert navigator.forgotten == 1

    # Not an error, not a step: the budget of one step was spent on the real decision.
    assert report["tool_errors"] == 0
    assert report["host_ignored_actions"] == 1
    assert report["steps_consumed"] == 1
    assert report["stop_reason"] == "model_text", "an ignored attempt must not exhaust the step budget"

    # The trace keeps the attempt, labelled, so a reader can see it; the chat model never does.
    trace = records(tmp_path / "run/tool-calls.jsonl")
    assert [(row["tool"], row.get("ignored", False)) for row in trace] == [("tap", True), ("analyze_screen", False)]
    assert trace[0]["ignored_because"].startswith("stale target")
    host_events = [m["content"] for m in requests[0]["messages"] if m["content"] and m["content"].startswith("Host-selected action")]
    assert len(host_events) == 1 and '"tool": "analyze_screen"' in host_events[0]
    assert "element_not_found" not in json.dumps(requests)


def test_a_genuine_failure_is_still_an_error(tmp_path):
    navigator = Navigator()

    async def execute(tool, arguments):
        return {"ok": False, "error": {"code": "timeout", "message": "the device did not answer"}}

    report, _, _ = run(tmp_path, navigator, execute, tools=TOOLS + [READ])
    assert report["tool_errors"] == 1 and report["host_ignored_actions"] == 0
    assert navigator.forgotten == 0
