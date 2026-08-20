"""Flow YAML parsing for new step kinds."""

from __future__ import annotations

from android_ui_analyser.flows import parse_flow_yaml


def test_parse_new_flow_steps() -> None:
    flow = parse_flow_yaml(
        """
name: ac_setup
steps:
  - dev_profile: ac
  - proxy_start
  - mock_replay: empty_inbox
  - flags_apply: experiments.yaml
  - a11y_scroll: { rid: list, direction: forward }
  - proxy_stop
  - dev_profile: default
"""
    )
    kinds = [s.kind for s in flow.steps]
    assert kinds == [
        "dev-profile",
        "proxy-start",
        "mock-replay",
        "flags-apply",
        "a11y-scroll",
        "proxy-stop",
        "dev-profile",
    ]
    assert flow.steps[0].arg == "ac"
    assert flow.steps[2].arg == "empty_inbox"
    assert flow.steps[4].resource_id == "list"
    assert flow.steps[4].arg == "forward"
