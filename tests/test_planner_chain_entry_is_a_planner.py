"""The planner chain must not call a method the entry may not have.

`ProviderFactory.create` is typed to the generic `Provider`, so `_capture_explain_llm` reached
`prov.decide(...)` unchecked, inside `except Exception: continue`. If a chain names something that
is not a planner — a misconfigured `planner.chain`, or a renamed method — the AttributeError is
swallowed and the narration silently returns nothing. That is exactly how the learned per-control
timing feature stayed dead from the commit that introduced it (see
`tests/test_learned_action_cost.py`), and a returned value cannot detect it: skipping the entry and
crashing on it both end with the same `None`.

So this asserts the *reach*, not the result: the probe raises a BaseException that neither
`except Exception` can swallow, and the test passes only when the engine never touches the
attribute at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.base import Availability, Provider
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config


class _ReachedTheProvider(BaseException):
    """Deliberately not an `Exception`, so the engine's blanket handlers cannot hide it."""


class _NotAPlanner(Provider):
    """A configured provider that never learned `decide` — the misconfiguration under test."""

    kind = "planner"
    name = "not-a-planner"

    def is_available(self) -> Availability:
        return Availability(True, "ok")

    def __getattr__(self, attribute: str) -> Any:
        raise _ReachedTheProvider(f"engine called {attribute!r} on a provider that lacks it")


class _ChainOfOne(ProviderFactory):
    """A planner chain whose single entry is not a planner."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.created: list[str] = []

    def is_enabled(self, kind: str) -> bool:
        return kind == "planner"

    def chain_names(self, kind: str) -> list[str]:
        return ["not-a-planner"] if kind == "planner" else []

    def create(self, kind: str, name: str) -> Provider:
        self.created.append(name)
        return _NotAPlanner()


def test_a_chain_entry_that_is_not_a_planner_is_skipped_not_called(tmp_path: Path) -> None:
    config = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    factory = _ChainOfOne(config)
    engine = Engine(config, device=FakeDevice(serial="emu-planner-chain"), factory=factory)

    narration = engine._capture_explain_llm({"narration": "tapped Continue", "summary": []})

    assert factory.created == ["not-a-planner"], "the chain entry must still be constructed"
    assert narration is None, "no planner produced text, so there is nothing to report"
