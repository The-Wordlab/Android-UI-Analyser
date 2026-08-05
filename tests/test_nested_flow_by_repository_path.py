"""A nested `flow:` step could only name a flow in AUA's own memory directory.

Observed while trying to factor shared preconditions into `flows/common/` and reference them from
`flows/derived/*`: `_run_steps` resolved a nested `flow:` reference **by name, from the memory
directory**, never as a path. A promoted flow that referenced a sibling therefore broke for anyone
whose memory directory did not happen to contain a flow of that name.

The cost is duplication that cannot be kept in step. Nine shared routes had to be **inlined** into
~35 derived flows, with the source named only in a description, so the same steps exist in many
copies and a fix to one does not propagate to the others. `grep` holds them together, which is a
convention rather than a guarantee.

Two things this deliberately does NOT do:

- It does not fall back to the name lookup when a path-looking reference resolves nowhere. The
  fallback would look up a *sanitised* spelling of the path in the memory directory
  (`common/auth.yaml` -> `common_auth.yaml`), where a chance match runs a different journey
  silently. Not finding the file the author named is recoverable; running somebody else's flow
  instead is not.
- It does not reinterpret a bare name. `flow: login` is a name today and stays one, so no existing
  flow changes meaning — which matters because this rewrites how every nested reference resolves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import looks_like_path, nested_flow_candidates
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config

SHARED = """
name: shared_auth
app: com.example.app
steps:
  - tap: "Sign in"
"""

DERIVED = """
name: derived_journey
app: com.example.app
steps:
  - flow: common/shared_auth.yaml
  - tap: "Apps"
"""


# --------------------------------------------------------------- what counts as a path


@pytest.mark.parametrize(
    "ref",
    [
        "common/shared_auth.yaml",
        "./shared_auth.yaml",
        "../common/shared_auth.yml",
        "shared_auth.yaml",
        "~/flows/shared_auth.yaml",
        "/abs/flows/shared_auth.yaml",
    ],
)
def test_a_reference_with_path_evidence_is_a_path(ref: str) -> None:
    assert looks_like_path(ref) is True


@pytest.mark.parametrize("ref", ["login", "shared_auth", "guest-setup", "world_cup_home", "", None])
def test_a_bare_word_stays_a_name(ref: str | None) -> None:
    """The asymmetry is deliberate: a name read as a path merely fails and says so, while a path
    read as a name can match an unrelated flow in the memory directory and run it."""
    assert looks_like_path(ref) is False  # type: ignore[arg-type]


# --------------------------------------------------------------- candidate precedence


def test_next_to_the_referring_flow_wins_over_the_collection_root(tmp_path: Path) -> None:
    """"Next to me" is the reading that makes a checked-in flow directory portable."""
    root = tmp_path / "flows"
    derived = root / "derived"
    derived.mkdir(parents=True)
    memory = tmp_path / "memory_flows"

    cands = nested_flow_candidates("common/shared_auth.yaml", derived, memory)

    assert cands[0] == derived / "common" / "shared_auth.yaml"
    assert cands[1] == root / "common" / "shared_auth.yaml"
    assert cands[-1] == memory / "common" / "shared_auth.yaml"


def test_the_memory_directory_cannot_shadow_a_repository_hit(tmp_path: Path) -> None:
    """Last, not first: what resolves inside the repo must not depend on one machine's install."""
    derived = tmp_path / "flows" / "derived"
    derived.mkdir(parents=True)
    cands = nested_flow_candidates("common/a.yaml", derived, tmp_path / "memory_flows")
    assert cands.index(tmp_path / "memory_flows" / "common" / "a.yaml") == len(cands) - 1


def test_an_absolute_reference_is_taken_as_given(tmp_path: Path) -> None:
    cands = nested_flow_candidates(str(tmp_path / "x.yaml"), tmp_path / "flows", tmp_path / "mem")
    assert cands == [tmp_path / "x.yaml"]


# --------------------------------------------------------------- end to end


SCREEN = (
    '<hierarchy rotation="0">'
    '<node class="android.widget.Button" package="com.example.app" text="Sign in"'
    ' resource-id="x:id/signIn" clickable="true" enabled="true" bounds="[40,300][1040,400]"/>'
    '<node class="android.widget.Button" package="com.example.app" text="Apps"'
    ' resource-id="x:id/navApps" clickable="true" enabled="true" bounds="[40,440][1040,540]"/>'
    "</hierarchy>"
)


def _engine(tmp_path: Path) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    dev = FakeDevice(hierarchy_xml=SCREEN, package="com.example.app")
    return Engine(cfg, device=dev, factory=ProviderFactory(cfg))


def _repo(tmp_path: Path) -> Path:
    """`flows/common/shared_auth.yaml` + `flows/derived/derived_journey.yaml`."""
    root = tmp_path / "flows"
    (root / "common").mkdir(parents=True)
    (root / "derived").mkdir(parents=True)
    (root / "common" / "shared_auth.yaml").write_text(SHARED, encoding="utf-8")
    (root / "derived" / "derived_journey.yaml").write_text(DERIVED, encoding="utf-8")
    return root


def test_a_derived_flow_runs_its_shared_precondition_from_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case that had to be inlined ~35 times: `flows/derived/*` referencing `flows/common/*`.

    Resolution walks up to the nearest enclosing `flows` directory, so `derived/` reaches
    `common/` without the author spelling `../`.
    """
    root = _repo(tmp_path)
    eng = _engine(tmp_path)

    out = eng.flow_run(file=str(root / "derived" / "derived_journey.yaml"))

    assert out["ok"] is True, out
    # Both flows' steps ran, in order: the shared precondition's tap, the `flow:` step that
    # pulled it in, then the derived flow's own tap.
    assert [s["step"] for s in out["steps_run"]] == [
        "tap 'Sign in'",
        "flow common/shared_auth.yaml",
        "tap 'Apps'",
    ], out


def test_a_path_that_resolves_nowhere_is_refused_rather_than_looked_up_by_name(
    tmp_path: Path,
) -> None:
    """The dangerous fallback, pinned shut.

    `common/shared_auth.yaml` sanitises to `common_shared_auth.yaml` for the memory-directory
    lookup. A chance match there would run an unrelated journey and report success.
    """
    eng = _engine(tmp_path)
    memory_flows = tmp_path / "home" / "flows"
    memory_flows.mkdir(parents=True)
    # The very file a name fallback would find.
    (memory_flows / "common_shared_auth.yaml.yaml").write_text(SHARED, encoding="utf-8")

    with pytest.raises(UsageError) as err:
        eng._resolve_nested_flow("common/shared_auth.yaml", tmp_path / "flows" / "derived")
    # The refusal has to say where it looked; StepFailure carries no message.
    assert "Tried:" in (err.value.hint or "")


def test_a_bare_name_still_resolves_from_the_memory_directory(tmp_path: Path) -> None:
    """No existing flow may change meaning — this is the compatibility half of the change."""
    eng = _engine(tmp_path)
    memory_flows = tmp_path / "home" / "flows"
    memory_flows.mkdir(parents=True)
    (memory_flows / "shared_auth.yaml").write_text(SHARED, encoding="utf-8")

    flow, base = eng._resolve_nested_flow("shared_auth", None)

    assert flow.name == "shared_auth"
    assert base == memory_flows


def test_a_sub_flows_own_relative_path_anchors_to_the_sub_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anchoring only the top-level flow left this one level down resolving against the daemon cwd.

    A shared precondition exists precisely to carry things like `flags_apply:`, so a nested flow
    whose own relative path did not resolve made the factored-out version useless even once it
    could be referenced.
    """
    root = tmp_path / "flows"
    (root / "common" / "flags").mkdir(parents=True)
    (root / "derived").mkdir(parents=True)
    (root / "common" / "flags" / "guest.yaml").write_text("flags: {}\n", encoding="utf-8")
    (root / "common" / "shared_auth.yaml").write_text(
        'name: shared_auth\napp: com.example.app\nsteps:\n  - flags_apply: flags/guest.yaml\n',
        encoding="utf-8",
    )
    (root / "derived" / "derived_journey.yaml").write_text(DERIVED, encoding="utf-8")

    eng = _engine(tmp_path)
    seen: list[str] = []
    monkeypatch.setattr(
        Engine, "flags_apply", lambda self, path, **kw: (seen.append(str(path)), {"ok": True})[1]
    )
    monkeypatch.setattr(Engine, "tap", lambda self, *a, **kw: ActionResult(ok=True, action="tap"))

    eng.flow_run(file=str(root / "derived" / "derived_journey.yaml"))

    assert seen == [str(root / "common" / "flags" / "guest.yaml")], seen


def test_nesting_depth_still_bounds_a_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Paths make a self-reference easy to write, so the existing depth cap must still catch it."""
    root = tmp_path / "flows"
    root.mkdir(parents=True)
    (root / "loop.yaml").write_text(
        'name: loop\napp: com.example.app\nsteps:\n  - flow: loop.yaml\n', encoding="utf-8"
    )
    eng = _engine(tmp_path)

    out = eng.flow_run(file=str(root / "loop.yaml"))

    assert out["ok"] is False, "an unbounded cycle would hang instead"
    assert out.get("code") == "unsupported_action", out
