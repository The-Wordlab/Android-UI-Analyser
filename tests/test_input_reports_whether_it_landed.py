"""`aua input` must not report success for text that never reached the field.

Seen repeatedly across a sweep: `input` returned `ok:true` on a field that still held what
it held before, and the lane read the empty result as *the product* ignoring it. This was
the last member of a family the rest of the tool has already closed off — `tap` re-analyzes,
`record` consults `ps`, `emulator stop` checks the running list.

The check is deliberately one-sided. It fails only when the field is readable, still holds
exactly its previous value, and does not contain what was typed. Plenty of fields
legitimately do not read back what you typed — `submit=True` sends the value and empties the
composer, password fields report a mask, and phone/date fields reformat as you type — so
those stay `ok` with `verified: None`. Reporting "unknown" honestly is the point; turning an
ambiguous read into a failure would be a worse bug than the one being fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config

_FIELD = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.EditText" text="{value}"
        resource-id="com.example.app:id/composer" clickable="true" enabled="true"
        focused="true" bounds="[40,600][1040,700]"/>
  <node index="1" class="android.widget.Button" text="Send"
        resource-id="com.example.app:id/send" clickable="true" enabled="true"
        bounds="[40,720][400,820]"/>
</hierarchy>"""


class _FieldDevice(FakeDevice):
    """A device whose focused field reads back a value the test chooses.

    `reads_back=None` models a field that cannot be read at all (no accessibility value,
    an offline read) — which must never be mistaken for an empty one.
    """

    def __init__(self, *, starts_with: str = "", reads_back: str | None, **kw: object) -> None:
        super().__init__(hierarchy_xml=_FIELD.format(value=starts_with), **kw)  # type: ignore[arg-type]
        self._reads_back = reads_back

    def focused_text(self) -> str | None:
        return self._reads_back


class _SubmitDevice(_FieldDevice):
    """A composer whose IME action either clears the field or leaves it populated."""

    def __init__(self, *, ime_clears: bool, **kw: object) -> None:
        super().__init__(starts_with="", reads_back="", **kw)
        self._ime_clears = ime_clears
        self._typed = ""

    def send_text(self, text: str, *, clear: bool = True) -> None:
        super().send_text(text, clear=clear)
        self._typed = text
        self._reads_back = text
        self._xml = _FIELD.format(value=text)

    def send_ime_action(self, action: str = "search") -> None:
        super().send_ime_action(action)
        if self._ime_clears:
            self._typed = ""
            self._reads_back = ""
            self._xml = _FIELD.format(value="")

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        if self._typed and y >= 700:
            self._typed = ""
            self._reads_back = ""
            self._xml = _FIELD.format(value="")


def _engine(dev: FakeDevice, tmp_path: Path) -> Engine:
    """An engine that has already analyzed, so element id 0 is the field."""
    eng = Engine(make_config(memory={"dir": str(tmp_path / "mem")}), device=dev)
    eng.analyze()
    return eng


def _flow_engine(dev: FakeDevice, tmp_path: Path) -> Engine:
    """`flow_run` analyzes for itself; a pre-analyze would hide a resolution bug."""
    return Engine(make_config(memory={"dir": str(tmp_path / "mem")}), device=dev)


def test_text_that_lands_is_verified(tmp_path: Path) -> None:
    dev = _FieldDevice(reads_back="hello world")
    out = _engine(dev, tmp_path).input_text(0, "hello world", observe=False)
    assert out.ok is True
    assert out.verified is True


def test_a_field_that_did_not_change_is_reported_as_a_failure(tmp_path: Path) -> None:
    """The reported bug: nothing was typed, and the old code said ok."""
    dev = _FieldDevice(starts_with="Ask anything", reads_back="Ask anything")
    out = _engine(dev, tmp_path).input_text(0, "hello world", observe=False)
    assert out.ok is False
    assert out.verified is False


def test_an_empty_field_that_stayed_empty_is_a_failure(tmp_path: Path) -> None:
    """The most common shape of it — an empty composer that swallowed the text."""
    dev = _FieldDevice(starts_with="", reads_back="")
    out = _engine(dev, tmp_path).input_text(0, "hello world", observe=False)
    assert out.ok is False
    assert out.verified is False


def test_submit_leaves_the_field_empty_and_that_is_not_a_failure(tmp_path: Path) -> None:
    """A chat composer is emptied *by sending*. Failing this would break the common case."""
    dev = _FieldDevice(starts_with="", reads_back="")
    out = _engine(dev, tmp_path).input_text(0, "hello world", submit=True, observe=False)
    assert out.ok is True
    assert out.verified is None


def test_ime_submit_that_leaves_text_in_the_composer_reports_not_submitted(
    tmp_path: Path,
) -> None:
    dev = _SubmitDevice(ime_clears=False)

    out = _engine(dev, tmp_path).input_text(0, "hello world", submit=True, observe=True)

    assert out.ok is True
    assert out.submitted is False
    assert out.recommended_call == {
        "kind": "semantic_send",
        "cli": "aua tap-and-analyze --rid send",
        "mcp": {"tool": "tap", "arguments": {"rid": "send"}},
        "reason": (
            "The IME action left the text in the composer and this is the unique visible "
            "send/submit/confirm control. Tap it without typing again."
        ),
        "executes": True,
    }
    assert "Do not type it again" in str(out.note)


def test_ime_submit_that_clears_the_composer_reports_submitted(tmp_path: Path) -> None:
    dev = _SubmitDevice(ime_clears=True)

    out = _engine(dev, tmp_path).input_text(0, "hello world", submit=True, observe=True)

    assert out.ok is True
    assert out.submitted is True
    assert out.recommended_call is None


def test_explicit_semantic_send_types_and_taps_in_one_top_level_call(tmp_path: Path) -> None:
    dev = _SubmitDevice(ime_clears=False)

    out = _engine(dev, tmp_path).input_text(
        0,
        "hello world",
        send_key="rid:send",
        observe=True,
    )

    assert out.ok is True
    assert out.action == "input-send"
    assert out.submitted is True
    assert not any(call == "send_ime_action" for call, _args in dev.calls)
    assert len([call for call, _args in dev.calls if call == "click"]) >= 2


def test_cli_explicit_send_reaches_the_shared_engine_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dev = _SubmitDevice(ime_clears=False)
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _engine, serial=None: dev)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_LEASE__REGISTRY_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_MEMORY__DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    result = CliRunner().invoke(
        app,
        ["input-and-analyze", "--rid", "composer", "hello", "--send", "rid:send"],
    )

    assert result.exit_code == 0, result.stdout + str(result.stderr or "")
    payload = json.loads(result.stdout)
    assert payload["action"] == "input-send"
    assert payload["submitted"] is True
    assert not any(call == "send_ime_action" for call, _args in dev.calls)


def test_cli_refuses_two_different_submission_mechanisms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dev = _SubmitDevice(ime_clears=False)
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _engine, serial=None: dev)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_LEASE__REGISTRY_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    result = CliRunner().invoke(
        app,
        [
            "input-and-analyze",
            "--rid",
            "composer",
            "hello",
            "--submit",
            "--send",
            "rid:send",
        ],
    )

    assert result.exit_code != 0
    assert "either --submit or --send" in result.stdout + str(result.stderr or "")


def test_an_unreadable_field_is_unknown_rather_than_unchanged(tmp_path: Path) -> None:
    """Never convert a failed observation into a verdict."""
    dev = _FieldDevice(starts_with="", reads_back=None)
    out = _engine(dev, tmp_path).input_text(0, "hello world", observe=False)
    assert out.ok is True
    assert out.verified is None


def test_a_masked_or_reformatted_value_stays_ok_and_unknown(tmp_path: Path) -> None:
    """A password mask and a reformatted phone number both changed — both really typed."""
    for read_back in ("•••••••••••", "+1 (555) 010-9999"):
        dev = _FieldDevice(starts_with="", reads_back=read_back)
        out = _engine(dev, tmp_path).input_text(0, "hello world", observe=False)
        assert out.ok is True, read_back
        assert out.verified is None, read_back


def test_typing_nothing_is_not_checked(tmp_path: Path) -> None:
    """`input ""` has no claim to verify, so it must not invent one."""
    dev = _FieldDevice(starts_with="whatever", reads_back="whatever")
    out = _engine(dev, tmp_path).input_text(0, "", observe=False)
    assert out.ok is True
    assert out.verified is None


def test_verified_is_absent_from_output_when_it_was_not_checked(tmp_path: Path) -> None:
    """`ok` must keep its meaning for every other action, so the field stays optional."""
    dev = _FieldDevice(starts_with="", reads_back=None)
    out = _engine(dev, tmp_path).input_text(0, "hi", observe=False)
    assert "verified" not in out.render("json")


def test_a_flow_step_that_typed_nothing_diverges(tmp_path: Path) -> None:
    """A flow must not carry on and blame the app for a screen its own input never reached."""
    flow = tmp_path / "type.yaml"
    flow.write_text(
        """
name: type_something
steps:
  - input: {id: composer, text: "hello world"}
  - tap: "Send"
""",
        encoding="utf-8",
    )
    dev = _FieldDevice(starts_with="Ask anything", reads_back="Ask anything")
    out = _flow_engine(dev, tmp_path).flow_run(file=str(flow))

    assert out["ok"] is False
    assert out["code"] == "input_not_applied"
    assert out["step_index"] == 0
    assert "--from-step 0" in out["hint"]
    # It stopped before the tap, rather than sending an empty message.
    assert not any(c for c, _ in dev.calls if c == "send_ime_action")
