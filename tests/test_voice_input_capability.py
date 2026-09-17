"""The emulator microphone is silent unless something injects into it.

Without this capability a voice scenario records silence, the app correctly reports that it heard
nothing, and that correct behaviour is read as a product failure -- which is exactly what happened
to `threads-send-voice-message-in-existing-thread` on 2026-09-17
(`docs/aua-deferred-fixes.md` item 85).
"""

import inspect

import pytest
from experiments.aua_controller.run_realapp import (
    CONTROLLER_CAPABILITIES,
    VOICE_INPUT_TOOL,
    export_primary_flow,
    realapp_tools,
)

from test_aua_controller_realapp import MCP_SCHEMAS as SCHEMAS


def _tools(*capabilities):
    return realapp_tools(SCHEMAS, capabilities=list(capabilities))


def _names(*capabilities):
    return [tool["function"]["name"] for tool in _tools(*capabilities)]


class TestTheToolToSpeak:
    def test_voice_input_is_a_known_capability(self) -> None:
        assert "voice-input" in CONTROLLER_CAPABILITIES

    def test_asking_for_it_offers_a_way_to_speak(self) -> None:
        assert VOICE_INPUT_TOOL in _names("voice-input")

    def test_a_run_that_does_not_ask_is_not_offered_it(self) -> None:
        assert VOICE_INPUT_TOOL not in _names()
        assert VOICE_INPUT_TOOL not in _names("app-lifecycle")

    def test_the_words_are_the_only_required_argument(self) -> None:
        """A scenario supplies its own script; every selector is optional."""
        tool = next(t for t in _tools("voice-input")
                    if t["function"]["name"] == VOICE_INPUT_TOOL)
        parameters = tool["function"]["parameters"]
        assert parameters["required"] == ["speech"]
        assert {"rid", "text", "desc", "control_mode"} <= parameters["properties"].keys()
        assert parameters["properties"]["control_mode"]["default"] == "hold"

    def test_speaking_counts_as_an_action_so_a_voice_route_can_be_saved(self) -> None:
        assert "VOICE_INPUT_TOOL" in inspect.getsource(export_primary_flow)

    def test_it_composes_with_the_other_capabilities(self) -> None:
        offered = _names("voice-input", "app-lifecycle")
        assert VOICE_INPUT_TOOL in offered
        assert "app_force_stop" in offered

    def test_an_unknown_capability_is_still_refused(self) -> None:
        with pytest.raises(Exception, match="unknown controller capabilities"):
            _tools("voice-input", "telepathy")
