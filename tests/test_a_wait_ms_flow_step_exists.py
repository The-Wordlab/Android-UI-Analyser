"""`wait_ms` — a bounded fixed-delay flow step.

Every other step kind either performs a UI action or waits on a UI condition (`wait_for`) or
UI quiescence (`wait_stable`). None of them can express "pause for a fixed interval regardless
of what the screen is doing" — the one thing a background flush needs. Measured consequence
(see the bug report): a step that writes app preferences via a deep link reports OK while the
write is flushed asynchronously on a background thread; the very next `stop_app`/`launch_app`
step kills the process before the flush lands, and the preferences are silently lost even
though every step reported success.

`wait_ms` closes that gap. Like every other agent-facing wait in this codebase, it is bounded
by `perf.max_wait_ms` (via `Engine._bounded_wait_ms`) rather than sleeping an arbitrary amount
— a flow cannot use it to block a caller indefinitely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import parse_flow_yaml, render_flow_yaml
from conftest import FakeDevice, make_config
from test_memory import HOME, P


def _engine(tmp_path: Path, device: FakeDevice | None = None) -> Engine:
    return Engine(
        make_config(
            memory={"enabled": True, "dir": str(tmp_path / "memory")},
            cache={"dir": str(tmp_path / "cache")},
            daemon={"enabled": False},
        ),
        device=device or FakeDevice(hierarchy_xml=HOME, package=P, serial="wait-ms"),
    )


class TestParsing:
    def test_a_bare_integer_is_the_delay_in_milliseconds(self) -> None:
        flow = parse_flow_yaml("steps:\n  - wait_ms: 500\n", name="w")
        step = flow.steps[0]
        assert step.kind == "wait-ms"
        assert step.timeout_ms == 500

    def test_the_mapping_form_is_equivalent(self) -> None:
        flow = parse_flow_yaml("steps:\n  - wait_ms: {timeout_ms: 500}\n", name="w")
        assert flow.steps[0].kind == "wait-ms"
        assert flow.steps[0].timeout_ms == 500

    def test_zero_is_refused(self) -> None:
        with pytest.raises(UsageError) as exc:
            parse_flow_yaml("steps:\n  - wait_ms: 0\n", name="w")
        assert "wait_ms" in str(exc.value)

    def test_negative_is_refused(self) -> None:
        with pytest.raises(UsageError):
            parse_flow_yaml("steps:\n  - wait_ms: -5\n", name="w")

    def test_a_missing_duration_is_refused(self) -> None:
        with pytest.raises(UsageError) as exc:
            parse_flow_yaml("steps:\n  - wait_ms: {}\n", name="w")
        assert "wait_ms" in str(exc.value)

    def test_is_not_a_bare_zero_arg_step(self) -> None:
        """Unlike `wait_stable`, `wait_ms` always needs a duration."""
        with pytest.raises(UsageError):
            parse_flow_yaml("steps:\n  - wait_ms\n", name="w")

    def test_unknown_kind_hint_lists_wait_ms(self) -> None:
        with pytest.raises(UsageError) as exc:
            parse_flow_yaml("steps:\n  - frobnicate: x\n", name="w")
        assert "wait_ms" in (exc.value.hint or "")


class TestRendering:
    def test_round_trips_through_render_as_a_bare_integer(self) -> None:
        flow = parse_flow_yaml("steps:\n  - wait_ms: 750\n", name="w")
        rendered = render_flow_yaml(flow)
        assert "wait_ms: 750" in rendered
        again = parse_flow_yaml(rendered, name="w")
        assert [s.model_dump() for s in again.steps] == [s.model_dump() for s in flow.steps]


class TestExecution:
    def test_a_flow_with_a_wait_ms_step_completes(self, tmp_path: Path) -> None:
        flow_path = tmp_path / "f.yaml"
        flow_path.write_text(f"name: f\napp: {P}\nsteps:\n  - wait_ms: 10\n", encoding="utf-8")

        result = _engine(tmp_path).flow_run(file=str(flow_path))

        assert result["ok"] is True

    def test_the_delay_is_bounded_by_the_existing_wait_ceiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `wait_ms` far past the ceiling must not sleep for the full requested amount.

        This is the "respect the existing wait-ceiling convention" requirement: `wait_ms`
        reuses `Engine._bounded_wait_ms` (backed by `perf.max_wait_ms`), the same mechanism
        every other wait already goes through, instead of sleeping an unbounded amount.
        """
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        engine = _engine(tmp_path)
        engine.config.perf.max_wait_ms = 1000  # tighten the ceiling for this test

        flow_path = tmp_path / "f.yaml"
        flow_path.write_text(
            f"name: f\napp: {P}\nsteps:\n  - wait_ms: 999999\n", encoding="utf-8"
        )
        result = engine.flow_run(file=str(flow_path))

        assert result["ok"] is True
        assert slept, "wait_ms did not sleep at all"
        assert all(seconds <= 1.0 for seconds in slept), slept
