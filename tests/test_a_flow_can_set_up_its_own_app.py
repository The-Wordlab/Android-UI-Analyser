"""A setup flow should not need a human to walk DevTools.

The flow library already replays UI journeys, applies feature flags, drives the proxy and
shapes the network. Two things a cold-start setup actually needs were missing, so every run
began with the same hand-walked preamble:

* **wiping app data** — the precondition for "cold run, cleared cache". A flow could stop and
  launch an app but never clear it, so the one step that makes a run reproducible was the one
  step that had to be done by hand.
* **writing to the app's own database** — `aua db execute` exists and is guarded, but flows
  could not reach it. Pointing a build at a different backend meant six taps through a
  settings screen instead of one row update.

Both are inherently destructive: unlike a tap, there is no label to inspect and no version of
them that is safe to replay speculatively. They are therefore destructive *by kind*, so
``goto`` can never wander into one while auto-replaying a learned route.
"""

from __future__ import annotations

import pytest
import yaml

from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import parse_flow_yaml
from android_ui_analyser.memory import RouteStep, is_destructive_step

LEXICON = ("delete", "remove", "sign out")


def _flow(steps: list) -> str:
    """The flow as an agent would write it: YAML on disk."""
    return yaml.safe_dump(
        {"schema_version": 1, "name": "setup", "app": "com.example.app", "steps": steps},
        sort_keys=False,
    )


class TestWipingAppDataIsAStep:
    def test_a_bare_clear_data_targets_the_flows_own_app(self) -> None:
        flow = parse_flow_yaml(_flow(["clear_data"]), name="setup")
        step = flow.steps[0]
        assert step.kind == "clear-data"
        # `arg` is where launch_app/stop_app already carry a package; empty means
        # "the journey's own app", so a bare step needs no target.
        assert not step.arg

    def test_clear_data_can_name_another_package(self) -> None:
        flow = parse_flow_yaml(_flow([{"clear_data": "com.other.app"}]), name="setup")
        assert flow.steps[0].arg == "com.other.app"

    def test_a_cold_setup_reads_as_one_flow(self) -> None:
        """The preamble that used to be typed by hand, as data."""
        flow = parse_flow_yaml(
            _flow(
                [
                    "clear_data",
                    {"launch_app": "com.example.app"},
                    {"wait_for": {"text": "Sign in", "timeout_ms": 5000}},
                ]
            ),
            name="setup",
        )
        assert [s.kind for s in flow.steps] == ["clear-data", "launch-app", "wait-for"]


class TestWritingTheAppsDatabaseIsAStep:
    def test_db_execute_carries_database_and_sql(self) -> None:
        flow = parse_flow_yaml(
            _flow(
                [
                    {
                        "db_execute": {
                            "database": "app.db",
                            "sql": "UPDATE settings SET value = 'staging' WHERE key = 'env'",
                        }
                    }
                ]
            ),
            name="setup",
        )
        step = flow.steps[0]
        assert step.kind == "db-execute"
        assert step.data["database"] == "app.db"
        assert step.data["sql"].startswith("UPDATE settings")

    def test_db_execute_defaults_to_the_flows_app(self) -> None:
        flow = parse_flow_yaml(
            _flow([{"db_execute": {"database": "app.db", "sql": "DELETE FROM cache"}}]),
            name="setup",
        )
        assert not flow.steps[0].arg

    def test_db_execute_without_sql_is_refused_at_parse_time(self) -> None:
        with pytest.raises(UsageError) as exc:
            parse_flow_yaml(_flow([{"db_execute": {"database": "app.db"}}]), name="setup")
        assert "sql" in str(exc.value).lower()

    def test_db_execute_without_a_database_is_refused_at_parse_time(self) -> None:
        with pytest.raises(UsageError) as exc:
            parse_flow_yaml(_flow([{"db_execute": {"sql": "DELETE FROM cache"}}]), name="setup")
        assert "database" in str(exc.value).lower()

    def test_a_bare_string_db_execute_is_refused(self) -> None:
        """`db_execute: "DELETE FROM x"` has nowhere to say which database."""
        with pytest.raises(UsageError):
            parse_flow_yaml(_flow([{"db_execute": "DELETE FROM cache"}]), name="setup")


class TestBothAreDestructiveByKindNotByLabel:
    def test_clear_data_is_destructive_with_no_label_to_inspect(self) -> None:
        step = RouteStep(kind="clear-data", arg="com.example.app")
        assert is_destructive_step(step, LEXICON) is True

    def test_db_execute_is_destructive_with_no_label_to_inspect(self) -> None:
        step = RouteStep(kind="db-execute", data={"database": "a.db", "sql": "DELETE FROM t"})
        assert is_destructive_step(step, LEXICON) is True

    def test_an_innocent_looking_tap_is_still_judged_on_its_label(self) -> None:
        """The label rule must survive: this change adds kinds, it does not replace the rule."""
        assert is_destructive_step(RouteStep(kind="tap", label="Continue"), LEXICON) is False
        assert is_destructive_step(RouteStep(kind="tap", label="Delete"), LEXICON) is True


class TestTheStepSurvivesARoundTrip:
    def test_rendering_a_parsed_flow_yields_the_same_steps(self) -> None:
        from android_ui_analyser.flows import render_flow_yaml

        original = _flow(
            [
                "clear_data",
                {"db_execute": {"database": "app.db", "sql": "UPDATE t SET a = 1"}},
            ]
        )
        flow = parse_flow_yaml(original, name="setup")
        reparsed = parse_flow_yaml(render_flow_yaml(flow), name="setup")
        assert [s.kind for s in reparsed.steps] == ["clear-data", "db-execute"]
        assert reparsed.steps[1].data["sql"] == "UPDATE t SET a = 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
