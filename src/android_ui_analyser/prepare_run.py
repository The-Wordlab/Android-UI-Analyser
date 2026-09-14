"""Run a prepared scenario - driven by AUA's own model when one is configured, or by the caller.

The expensive part of an agent-driven run is not the device, it is the round trips: every tap,
every screen read and every judgement crosses back into the calling agent's context.  When a
controller model is configured, AUA runs that loop itself against the contract the two of them
agreed, and the caller pays for one question and one answer.

When one is not configured this returns the ordered commands instead, and the caller drives.  That
is not a degraded mode - it is how AUA has always worked, and the contract, the setup plan and the
evidence bundle are identical either way.  What must never happen is ambiguity about which of the
two you got, so the result always says, and says why.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import Config
from .errors import UsageError
from .evidence import handback

CONTROLLER_RELATIVE = Path("experiments") / "aua_controller" / "run_realapp.py"


def bundled_controller(start: Path | None = None) -> Path | None:
    """The controller shipped beside this source tree, if this is a source checkout.

    An installed wheel does not carry ``experiments/``, so this legitimately returns None there;
    that is a reason to report "no driver", not to guess at a path that does not exist.
    """

    base = (start or Path(__file__).resolve()).parent
    for parent in [base, *base.parents][:5]:
        candidate = parent / CONTROLLER_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def controller_state(cfg: Config, *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Whether AUA can drive this itself, and - when it cannot - exactly what is missing."""

    env = os.environ if environ is None else environ
    controller = cfg.controller
    command: list[str] = list(controller.command)
    if not command:
        script = bundled_controller()
        command = [sys.executable, str(script)] if script is not None else []
    state: dict[str, Any] = {
        "enabled": controller.enabled,
        "model": controller.model,
        "judge_model": controller.judge_model,
        "api_key_env": controller.api_key_env,
        "command": command,
    }
    if not controller.enabled:
        state["available"] = False
        state["reason"] = (
            "controller.enabled is false; set it to run scenarios through AUA's own model"
        )
    elif not command:
        state["available"] = False
        state["reason"] = (
            "no controller found: this AUA was installed without `experiments/`, and "
            "controller.command is empty"
        )
    elif not str(env.get(controller.api_key_env, "")).strip():
        state["available"] = False
        state["reason"] = f"{controller.api_key_env} is not set in this environment"
    else:
        state["available"] = True
        state["reason"] = "ready"
    return state


def harness_command(
    cfg: Config, scenario: Mapping[str, Any], *, output: Path, command: Sequence[str]
) -> list[str]:
    """The exact controller invocation for one saved scenario."""

    controller = cfg.controller
    answers = scenario.get("answers") or {}
    argv = [
        *command,
        "--goal",
        str(scenario["goal"]),
        "--package",
        str(scenario["package"]),
        "--output",
        str(output),
        "--contract",
        str(scenario["contract"]),
        "--model",
        controller.model,
        "--judge-model",
        controller.judge_model,
        "--base-url",
        controller.base_url,
        "--api-key-env",
        controller.api_key_env,
        "--max-steps",
        str(controller.max_steps),
        "--time-limit",
        str(controller.time_limit_s),
        "--cost-limit-usd",
        str(controller.cost_limit_usd),
        "--judge-cost-limit-usd",
        str(controller.judge_cost_limit_usd),
    ]
    for fallback in controller.judge_fallbacks:
        argv += ["--judge-fallback", str(fallback)]
    from .prepare import listed_answer

    for flag in listed_answer(answers.get("flags")):
        argv += ["--flags", flag]
    for flow in listed_answer(answers.get("setup_flow")):
        argv += ["--setup-flow", flow]
    build = str(answers.get("build") or "").strip()
    if build:
        argv += ["--apk", build]
    # Only the strategy that means "really start from nothing" wipes the app; every other
    # pre-condition was agreed as something AUA sets and puts back.
    if answers.get("seeding") == "reinstall":
        argv.append("--fresh")
    if controller.record:
        argv.append("--record")
    return argv


def child_environment(command: Sequence[str], environ: Mapping[str, str]) -> dict[str, str]:
    """Let the controller import this AUA, not whichever one happens to be on the path."""

    env = dict(environ)
    root = Path(command[-1]).resolve().parents[2] if len(command) > 1 else None
    if root is not None and (root / "src").is_dir():
        existing = env.get("PYTHONPATH", "")
        paths = [str(root / "src"), str(root)]
        env["PYTHONPATH"] = os.pathsep.join([*paths, existing] if existing else paths)
    return env


def manual_plan(scenario: Mapping[str, Any], *, output: Path, reason: str) -> dict[str, Any]:
    """What to run when AUA is not driving: the same contract, executed by the caller."""

    goal = str(scenario["goal"])
    commands = [
        "aua session start"
        f" --goal {goal!r}"
        f" --contract {scenario['contract']}"
        f" --app {scenario['package']}"
        f" --artifacts-dir {output} --evidence all"
    ]
    answers = scenario.get("answers") or {}
    build = str(answers.get("build") or "").strip()
    if build:
        commands[0] += f" --apk {build}"
        if answers.get("seeding") == "reinstall":
            commands[0] += " --fresh --yes"
    commands.append("aua record start")
    commands.append("# ...drive to the goal; each checkpoint completes from fresh assertions...")
    commands.append("aua record stop")
    commands.append("aua session finish --full")
    return {
        "ok": True,
        "driven_by": "caller",
        "why": reason,
        "scenario": scenario.get("name"),
        "goal": goal,
        "package": scenario.get("package"),
        "contract": scenario.get("contract"),
        "setup": scenario.get("setup", []),
        "artifacts_dir": str(output),
        "commands": commands,
    }


# The controller reports its verdict as a small object - `{"oracle": ..., "verified": ...,
# "verdict": "pass", "reasons": [...]}` - not a bare string. Reading it as one crashed the handback
# with `unhashable type: 'dict'` after a full, successful device run, which is the worst possible
# place to lose a result.
PASSING_VERDICTS = frozenset({"pass", "passed", "pass_with_warning", "passed_with_warning"})


def verdict_of(result: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The verdict word and the reasons behind it, whichever shape the controller used."""

    raw = result.get("verdict")
    if isinstance(raw, Mapping):
        reasons = raw.get("reasons")
        return (
            str(raw.get("verdict") or "unverified"),
            [str(reason) for reason in reasons] if isinstance(reasons, list) else [],
        )
    return (str(raw or "unverified"), [])


def run_scenario(
    cfg: Config,
    scenario: Mapping[str, Any],
    *,
    output: str | Path,
    environ: Mapping[str, str] | None = None,
    execute: Any = subprocess.run,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Drive the scenario, or return the plan that drives it.  The result always says which."""

    import json

    env = dict(os.environ if environ is None else environ)
    target = Path(output).expanduser()
    state = controller_state(cfg, environ=env)
    if not state["available"]:
        return manual_plan(scenario, output=target, reason=state["reason"])

    target.mkdir(parents=True, exist_ok=True)
    argv = harness_command(cfg, scenario, output=target, command=state["command"])
    completed = execute(
        argv,
        cwd=str(target),
        env=child_environment(state["command"], env),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    result_path = target / "result.json"
    result: dict[str, Any]
    try:
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        result = loaded if isinstance(loaded, dict) else {"verdict": "blocked"}
    except (OSError, ValueError):
        # No result file means the controller never reached a judgement. Saying "failed" here
        # would report a product verdict for an infrastructure problem, which is the one mistake
        # a QA harness must not make.
        result = {
            "verdict": "blocked",
            "reason": "the controller produced no result.json",
            "exit_code": getattr(completed, "returncode", None),
            "stderr_tail": (getattr(completed, "stderr", "") or "")[-2000:],
        }
    payload = handback(
        result,
        root=target,
        extra=[target / "journey.mp4"],
        report=target / "verdict.md",
    )
    verdict, reasons = verdict_of(result)
    payload["verdict"] = verdict
    payload["ok"] = verdict in PASSING_VERDICTS
    if reasons:
        payload["reasons"] = reasons
    payload["driven_by"] = "aua"
    payload["model"] = cfg.controller.model
    payload["scenario"] = scenario.get("name")
    payload["goal"] = scenario.get("goal")
    payload["contract"] = scenario.get("contract")
    payload["artifacts_dir"] = str(target)
    payload["exit_code"] = getattr(completed, "returncode", None)
    return payload


def default_output(scenario_name: str, *, base: str | Path | None = None) -> Path:
    """Somewhere predictable to put a run's evidence when the caller does not choose."""

    root = Path(base).expanduser() if base else Path.cwd() / ".aua-runs"
    return root / scenario_name


def require_adb() -> None:
    """A run needs a device toolchain; say so before spending a model call finding out."""

    if shutil.which("adb") is None:
        raise UsageError(
            "adb is not on PATH, so no scenario can run",
            hint="Install the Android platform-tools, or run the returned commands elsewhere.",
        )
