"""Drive one unauthored goal on a real application, then judge it from the frames.

The fixture runner needs an authored contract to accept ``session_finish``. A real
application usually has none, so the model's completion claim is accepted as a *signal*
that stops the loop, and a separate bounded judgement (``judgement.py``) decides the
outcome from the observed frames. Screens seen along the way can be named for the map.

Everything the model reads is compacted (``compaction.py``) after the hosted privacy
projection. Raw evidence stays untouched under ``<output>/controller/evidence``.

The runner is application-agnostic: goal, package and optional setup flow come from the
caller. Paid model calls happen only through the injected sender; the result names the
model, provider and reported cost of every tier that ran.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from experiments.aua_controller.agent_loop import run_agent
from experiments.aua_controller.compaction import FrameCompactor
from experiments.aua_controller.hosted import BACKENDS, validate_endpoint, validate_request_config
from experiments.aua_controller.hosted_projection import hosted_model_view
from experiments.aua_controller.judgement import (
    Decider,
    ScreenNamer,
    judge_outcome_votes,
    summarize_route,
)
from experiments.aua_controller.run_live import (
    COMPACT_SYSTEM,
    SYSTEM,
    RunError,
    _error_text,
    compact_schema,
    tool_result,
)
from experiments.aua_controller.session_state import observation_frame

FORMAT = "aua-realapp-run-v1"
CONTROLLER_TOOLS = (
    "analyze_screen", "tap_and_analyze", "input_and_analyze", "swipe_and_analyze",
    "wait_and_analyze", "key_and_analyze", "session_progress", "session_finish",
)
FINISH_OUTCOMES = ("achieved", "already_satisfied", "blocked", "not_achievable")
REALAPP_SYSTEM = """
Real-application mode. There is no authored checklist; you decide when the goal is met.
If the initial observation already shows the requested end state, call session_finish at once
with outcome "already_satisfied". After you observe the requested end state, call session_finish
once with outcome "achieved" and a one-line note; do not spend steps collecting extra evidence.
If a login wall, permission prompt, network failure or missing precondition stops you, call
session_finish with outcome "blocked" and say what blocked you. If the app cannot do what is
asked, use "not_achievable". A separate reviewer verifies your claim from the screens.
"""


def finish_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": list(FINISH_OUTCOMES)},
            "note": {"type": "string", "maxLength": 240},
        },
        "required": ["outcome"],
        "additionalProperties": False,
    }


def realapp_tools(schemas: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """compact-v1 tools, with session_finish carrying the model's outcome claim."""
    tools = []
    for name in CONTROLLER_TOOLS:
        if name not in schemas:
            raise RunError(f"AUA MCP does not offer {name}")
        parameters = finish_schema() if name == "session_finish" else compact_schema(name, schemas[name])
        description = str(schemas[name].get("description") or "")[:300]
        if name == "session_finish":
            description = "Claim the goal is finished (or blocked) with an outcome and a short note."
        tools.append({"type": "function", "function": {"name": name, "description": description,
                                                       "parameters": parameters}})
    return tools


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def controller_cost(report: dict[str, Any]) -> float:
    accounting = report.get("cost_accounting") or {}
    if isinstance(accounting.get("reported_usd"), (int, float)):
        return float(accounting["reported_usd"])
    return sum(float((usage or {}).get("cost") or 0) for usage in report.get("usage", []))


def verdict_markdown(result: dict[str, Any]) -> str:
    verdict = result["verdict"]
    lines = [f"# {verdict['verdict'].upper()}  ·  {result['goal']}", ""]
    stop = (result.get("controller") or {}).get("stop_reason")
    lines.append(f"Oracle: `{verdict['oracle']}` · verified: {verdict['verified']} · controller stop: `{stop}`")
    if result.get("error"):
        lines.append(f"Run error: {result['error']}")
    claim = result.get("claim")
    if claim:
        lines.append(f"Controller claim: **{claim.get('outcome')}** — {claim.get('note') or ''}".rstrip(" —"))
    lines.append("")
    for reason in verdict.get("reasons", []):
        lines.append(f"- {reason}")
    cost = result["cost"]
    lines += ["", "| tier | model | provider | reported USD |", "|---|---|---|---|"]
    for tier in ("controller", "judge", "map"):
        entry = cost.get(tier)
        if entry:
            lines.append(f"| {tier} | {entry.get('model')} | {entry.get('provider')} | {entry.get('usd'):.6f} |")
    lines.append(f"| total | | | {cost['total_usd']:.6f} |")
    screens = result.get("screens") or []
    if screens:
        lines += ["", "## Screens", ""]
        for screen in screens:
            lines.append(f"- `{screen['logical_name']}` ({screen['kind']}): {screen['purpose']}")
    return "\n".join(lines) + "\n"


async def run_realapp(
    *,
    call_tool: Callable[[str, dict[str, Any]], Awaitable[Any]],
    list_tools: Callable[[], Awaitable[dict[str, dict[str, Any]]]],
    send: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    goal: str,
    package: str,
    output: Path,
    model: str,
    request_config: dict[str, Any] | None = None,
    backend: str = "openrouter",
    launch: bool = False,
    activity: str | None = None,
    setup_flow_yaml: str | None = None,
    judge: bool = True,
    judge_votes: int = 2,
    name_screens: bool = False,
    max_named_screens: int = 8,
    max_steps: int = 24,
    time_limit_s: float = 300,
    max_tokens: int = 4096,
    max_request_bytes: int = 200_000,
    cost_limit_usd: float = 0.05,
    judge_cost_limit_usd: float = 0.02,
    judge_max_tokens: int = 1024,
    terminal_claim_limit: int = 1,
    no_progress_limit: int = 4,
    max_elements: int = 60,
    request_timeout_s: float = 90,
) -> dict[str, Any]:
    """Return the run result; also written to ``<output>/result.json`` and ``verdict.md``."""
    if backend not in BACKENDS:
        raise RunError("unknown backend")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RunError("real-app output directory must be empty")
    settings = validate_request_config(request_config or {}) if backend == "openrouter" else copy.deepcopy(request_config or {})
    started = time.monotonic()
    result: dict[str, Any] = {
        "format": FORMAT, "goal": goal, "package": package, "model": model, "backend": backend,
        "request_config": settings, "session_id": None, "serial": None, "setup": [],
        "controller": None, "claim": None, "verdict": None, "screens": [], "route": None,
        "cost": {"total_usd": 0.0}, "error": None, "report_is_untrusted": True,
    }
    setup_log = output / "setup-calls.jsonl"

    async def call(name: str, arguments: dict[str, Any], actor: str) -> dict[str, Any]:
        tick = time.monotonic()
        record: dict[str, Any] = {"actor": actor, "tool": name, "arguments": arguments}
        try:
            decoded = tool_result(await call_tool(name, arguments))
            record["ok"] = decoded.get("ok")
            return decoded
        except Exception as exc:
            record["error"] = _error_text(exc)
            raise
        finally:
            record["duration_ms"] = (time.monotonic() - tick) * 1000
            with setup_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    session_id: str | None = None
    try:
        start = await call("session_start", {"goal": goal, "package": package, "headed": False,
                                             "artifacts_dir": str((output / "aua").resolve()),
                                             "evidence": "all"}, "setup")
        session_id = start.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RunError("session_start returned no session_id")
        result["session_id"], result["serial"] = session_id, start.get("serial")
        if launch:
            arguments: dict[str, Any] = {"package": package}
            if activity:
                arguments["activity"] = activity
            await call("app_launch_and_analyze", arguments, "setup")
        if setup_flow_yaml:
            flow = await call("flow_run", {"yaml": setup_flow_yaml, "assist": False}, "setup")
            result["setup"].append({"flow_run_ok": flow.get("ok"), "error": flow.get("error")})
            if flow.get("ok") is False:
                raise RunError("setup flow failed: " + json.dumps(flow.get("error"))[:300])
        initial = await call("analyze_screen", {"source": "hierarchy", "no_cache": True}, "setup")
        if observation_frame(initial) is None:
            raise RunError("initial analyze_screen returned no fresh frame")
        schemas = await list_tools()
        tools = realapp_tools(schemas)
        claims: list[dict[str, Any]] = []

        async def controller_call(name: str, arguments: dict[str, Any]) -> Any:
            if name == "session_finish":
                claims.append(copy.deepcopy(arguments))
                arguments = {"session_id": session_id, "allow_incomplete": False, "summary": False}
            elif name == "session_progress":
                arguments = {"session_id": session_id}
            return await call_tool(name, arguments)

        compactor = FrameCompactor(max_elements=max_elements)
        report = await run_agent(
            send=send, call_tool=controller_call, tools=tools,
            system_prompt=SYSTEM + COMPACT_SYSTEM + REALAPP_SYSTEM, user_prompt="Goal: " + goal,
            initial_observation=initial, model=model, output=output / "controller",
            request_config=settings, backend=backend, max_tokens=max_tokens, max_steps=max_steps,
            time_limit_s=time_limit_s, max_request_bytes=max_request_bytes,
            cost_limit_usd=cost_limit_usd, observation_filter=hosted_model_view,
            model_observation_filter=compactor, request_timeout_s=request_timeout_s,
            terminal_tools=frozenset({"session_finish"}), terminal_claim_limit=terminal_claim_limit,
            no_progress_limit=no_progress_limit,
        )
        result["controller"] = {
            key: report.get(key) for key in (
                "stop_reason", "error", "steps_consumed", "model_requests", "tool_calls_executed",
                "tool_errors", "schema_repairs", "terminal_claims", "no_progress_streak",
                "returned_models", "providers", "model_http_seconds", "tool_seconds", "duration_seconds",
                "final_model_text",
            )
        }
        result["controller"]["prompt_tokens"] = [
            (usage or {}).get("prompt_tokens") for usage in report.get("usage", [])]
        result["controller"]["compaction"] = {"frames": compactor.frames_seen, "unchanged_hits": compactor.unchanged_hits}
        result["claim"] = claims[-1] if claims else None
        result["cost"]["controller"] = {
            "model": (report.get("returned_models") or [model])[0],
            "provider": (report.get("providers") or [None])[0], "usd": controller_cost(report),
        }

        final = await call("analyze_screen", {"source": "hierarchy", "no_cache": True}, "final")
        # Kept raw so a verdict can be re-judged offline from the same evidence later.
        (output / "final-observation.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
        frames = []
        for entry in report.get("evidence", []):
            raw = json.loads((output / "controller" / entry["path"]).read_text(encoding="utf-8"))
            if observation_frame(raw) is not None:
                frames.append({"ref": entry["ref"], "tool": entry.get("tool"), "raw": raw})
        actions = [
            {"step": call_record["step"], "tool": call_record["tool"], "arguments": call_record.get("arguments")}
            for call_record in _load_jsonl(output / "controller" / "tool-calls.jsonl")
            if call_record.get("executed") is True
        ]
        stop = report.get("stop_reason")
        if stop == "terminal_tool":
            result["verdict"] = {"oracle": "aua_session_contract", "verified": True, "verdict": "pass",
                                 "reasons": ["AUA accepted session_finish against its own contract."]}
        elif stop in ("terminal_claimed", "model_text", "no_progress") and judge:
            decider = Decider(send, model=model, backend=backend, request_config=settings,
                              max_tokens=judge_max_tokens, cost_limit_usd=judge_cost_limit_usd,
                              output=output / "judge")
            # AUA's goal_progress without an authored contract is always 0/1 active; it would
            # only mislead a judge, so real-app mode does not pass it.
            context_actions = list(actions)
            if result["claim"]:
                context_actions.append({"step": len(actions), "tool": "session_finish",
                                        "arguments": {"controller_claim_untrusted": result["claim"]}})
            verdict = await judge_outcome_votes(
                decider, votes=judge_votes, goal=goal, final_frame=final,
                frames=[frame["raw"] for frame in frames[-4:-1]] if len(frames) > 1 else [],
                actions=context_actions,
            )
            if stop == "no_progress" and verdict["verdict"] == "pass":
                verdict["verdict"] = "pass_with_warning"
                verdict["reasons"].insert(0, "Controller stalled on an unchanged screen before finishing.")
            verdict["controller_stop_reason"] = stop
            result["verdict"] = verdict
            result["cost"]["judge"] = {"model": model, "provider": (verdict["votes"][0].get("provider") if verdict["votes"] else None),
                                       "usd": verdict["cost"], "decider": decider.report()}
        else:
            reason = report.get("error") or f"controller stopped with {stop}"
            result["verdict"] = {"oracle": "none", "verified": False, "verdict": "unverified",
                                 "reasons": [str(reason)[:300]], "controller_stop_reason": stop}
        if name_screens:
            namer_decider = Decider(send, model=model, backend=backend, request_config=settings,
                                    max_tokens=512, cost_limit_usd=judge_cost_limit_usd, output=output / "map")
            namer = ScreenNamer(namer_decider)
            ordered = [{"ref": "initial", "tool": None, "raw": initial}] + frames + [{"ref": "final", "tool": None, "raw": final}]
            transitions: list[dict[str, Any]] = []
            previous_name: str | None = None
            for frame in ordered:
                if len(namer.distinct()) >= max_named_screens and namer.fingerprint(frame["raw"]) not in namer.by_fingerprint:
                    transitions.append({"after": frame["tool"], "to": "unnamed(limit)"})
                    continue
                meta = (observation_frame(frame["raw"]) or {}).get("meta") or {}
                known = meta.get("known_screen") if isinstance(meta.get("known_screen"), str) else None
                entry = await namer.name(frame["raw"], known_name=known)
                name = entry["logical_name"] if entry else None
                if name and name != previous_name:
                    transitions.append({"after": frame["tool"], "from": previous_name, "to": name, "evidence_ref": frame["ref"]})
                    previous_name = name
            result["screens"] = namer.distinct()
            if result["screens"]:
                result["route"] = await summarize_route(namer_decider, goal=goal, screens=result["screens"],
                                                        transitions=transitions)
                result["route"]["transitions"] = transitions
            result["cost"]["map"] = {"model": model, "provider": (result["cost"]["controller"] or {}).get("provider"),
                                     "usd": namer_decider.total_cost, "decider": namer_decider.report()}
            (output / "screens.json").write_text(json.dumps(result["screens"], ensure_ascii=False, indent=2) + "\n")
            if result["route"]:
                (output / "route.json").write_text(json.dumps(result["route"], ensure_ascii=False, indent=2) + "\n")
    except Exception as exc:
        result["error"] = _error_text(exc)
        if result["verdict"] is None:
            result["verdict"] = {"oracle": "none", "verified": False, "verdict": "unverified",
                                 "reasons": [result["error"][:300]]}
    finally:
        if session_id:
            try:
                await call("session_finish", {"session_id": session_id, "allow_incomplete": True, "summary": False}, "cleanup")
            except Exception as exc:  # cleanup failure is recorded, never masks the run
                result["cleanup_error"] = _error_text(exc)
        result["cost"]["total_usd"] = round(sum(
            float(entry["usd"]) for key, entry in result["cost"].items()
            if key != "total_usd" and isinstance(entry, dict)), 8)
        result["duration_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
        (output / "verdict.md").write_text(verdict_markdown(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goal", required=True)
    parser.add_argument("--package", required=True, help="Application package under test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("openrouter-comparison.json"))
    parser.add_argument("--model", required=True, help="Candidate id or repository from the manifest")
    parser.add_argument("--provider", help="Override the pinned provider slug (for example when a pin rots)")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPEN_ROUTER_API_KEY")
    parser.add_argument("--aua-command", default="aua")
    parser.add_argument("--launch", action="store_true", help="app_launch_and_analyze before the goal")
    parser.add_argument("--activity")
    parser.add_argument("--setup-flow", type=Path, help="AUA flow YAML run before the goal (login, reset)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-votes", type=int, default=2, choices=[1, 2])
    parser.add_argument("--map", action="store_true", help="Name screens and summarise the route (paid)")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--cost-limit-usd", type=float, default=0.05)
    parser.add_argument("--judge-cost-limit-usd", type=float, default=0.02)
    parser.add_argument("--terminal-claim-limit", type=int, default=1)
    parser.add_argument("--no-progress-limit", type=int, default=4)
    parser.add_argument("--max-elements", type=int, default=60)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    candidate = next((item for item in manifest["models"] if args.model in {item["id"], item["repository"]}), None)
    if candidate is None:
        parser.error("model must be a candidate in the manifest")
    request_config = copy.deepcopy(candidate.get("request_config", {}))
    if args.provider:
        request_config.setdefault("provider", {})
        request_config["provider"]["only"] = [args.provider]
        request_config["provider"]["order"] = [args.provider]
    request_config = validate_request_config(request_config)
    key = os.environ.get(args.api_key_env)
    validate_endpoint(args.base_url, key)

    async def execute() -> dict[str, Any]:
        import httpx
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        headers = {"Authorization": f"Bearer {key}"}
        server = StdioServerParameters(command=args.aua_command, args=["mcp"])
        async with httpx.AsyncClient(headers=headers, timeout=120, follow_redirects=False) as http:
            async def send(payload: dict[str, Any]) -> dict[str, Any]:
                response = await http.post(args.base_url.rstrip("/") + "/chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise RunError("endpoint returned non-object JSON")
                return body

            async with (
                stdio_client(server) as (read, write),
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=180)) as session,
            ):
                await session.initialize()

                async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                    return await session.call_tool(name, arguments)

                async def list_tools() -> dict[str, dict[str, Any]]:
                    listing = await session.list_tools()
                    schemas = {}
                    for tool in listing.tools:
                        data = tool.model_dump(mode="json")
                        schema = dict(data["inputSchema"])
                        schema.setdefault("description", data.get("description"))
                        schemas[data["name"]] = schema
                    return schemas

                return await run_realapp(
                    call_tool=call_tool, list_tools=list_tools, send=send, goal=args.goal,
                    package=args.package, output=args.output.resolve(), model=candidate["repository"],
                    request_config=request_config, launch=args.launch, activity=args.activity,
                    setup_flow_yaml=args.setup_flow.read_text() if args.setup_flow else None,
                    judge=not args.no_judge, judge_votes=args.judge_votes, name_screens=args.map,
                    max_steps=args.max_steps, time_limit_s=args.time_limit, max_tokens=args.max_tokens,
                    cost_limit_usd=args.cost_limit_usd, judge_cost_limit_usd=args.judge_cost_limit_usd,
                    terminal_claim_limit=args.terminal_claim_limit, no_progress_limit=args.no_progress_limit,
                    max_elements=args.max_elements,
                )

    result = asyncio.run(execute())
    print(json.dumps({
        "verdict": result["verdict"]["verdict"], "oracle": result["verdict"]["oracle"],
        "stop_reason": (result["controller"] or {}).get("stop_reason"), "claim": result.get("claim"),
        "steps": (result["controller"] or {}).get("steps_consumed"),
        "total_usd": result["cost"]["total_usd"], "error": result["error"], "output": str(args.output),
    }, default=str))
    return 0 if result["verdict"]["verdict"] in {"pass", "pass_with_warning"} else 1


if __name__ == "__main__":
    sys.exit(main())
