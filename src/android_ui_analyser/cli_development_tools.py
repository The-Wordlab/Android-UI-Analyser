"""Thin CLI surfaces for optional development tools, previews and authored journeys."""

from __future__ import annotations

import json
from typing import Any

import typer

from .errors import ExitCode, UsageError


def _json(value: str, expected: type) -> Any:
    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise UsageError(f"invalid JSON: {exc}") from None
    if not isinstance(parsed, expected):
        raise UsageError(f"expected a JSON {expected.__name__}")
    return parsed


def register(app: typer.Typer, run: Any, route: Any, emit: Any) -> None:
    """Receive shared CLI plumbing without importing the CLI module recursively."""
    tools = typer.Typer(help="Optional platform developer tools: setup, discover, run.")
    previews = typer.Typer(help="Render and inspect Compose previews without a device.")
    profile = typer.Typer(help="Profile the leased target with durable cleanup.")
    journeys = typer.Typer(help="Authored agent journeys with assertions and explicit visual reviews.")
    app.add_typer(tools, name="devtools")
    app.add_typer(previews, name="preview")
    app.add_typer(profile, name="profile")
    app.add_typer(journeys, name="journey")

    def emit_result(result: Any, fmt: Any, method: str) -> None:
        emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            assertion = method in {"preview_check", "journey_check", "journey_record", "journey_finish"}
            raise typer.Exit(int(ExitCode.ASSERTION if assertion else ExitCode.DEVICE))

    def call(ctx: typer.Context, method: str, **kwargs: Any) -> None:
        run(ctx, lambda engine, fmt: emit_result(route(engine, method, **kwargs), fmt, method))

    @tools.command("status")
    def status(ctx: typer.Context) -> None:
        """Find the optional tool and report its version; never install implicitly."""
        call(ctx, "development_tools_status")

    @tools.command("setup")
    def setup(ctx: typer.Context) -> None:
        """Install the optional tool for this user from its official distribution."""
        call(ctx, "development_tools_setup")

    @tools.command("catalogue")
    def catalogue(ctx: typer.Context, operation: str | None = typer.Argument(None)) -> None:
        """List operations or inspect one operation's accepted arguments/options."""
        call(ctx, "development_tools_catalogue", operation=operation)

    @tools.command("run")
    def execute(
        ctx: typer.Context,
        operation: str = typer.Argument(..., help="Dotted operation from devtools catalogue."),
        arguments: list[str] | None = typer.Argument(None, help="Positional values, never shell code."),
        options: str = typer.Option("{}", "--options", help="JSON object; option names omit --."),
        project: str | None = typer.Option(None, "--project", help="Working directory for project tools."),
        timeout: float = typer.Option(120, "--timeout", min=1, max=3600),
        yes: bool = typer.Option(False, "--yes", help="Confirm a catalogued destructive/admin operation."),
        profile_kind: str | None = typer.Option(None, "--profile-kind", help="Owned trace or heap session; only for supported profiling operations."),
    ) -> None:
        """Run a catalogued operation; host tools need no device, target tools use its lease."""
        def go(engine: Any, fmt: Any) -> None:
            emit(route(
                engine, "development_tool", operation=operation, arguments=arguments or [],
                options=_json(options, dict), project_dir=project, timeout_s=timeout, confirmed=yes,
                profile_kind=profile_kind,
            ), fmt)
        run(ctx, go)

    @previews.command("list")
    def preview_list(
        ctx: typer.Context, reference: str | None = typer.Argument(None),
        project: str | None = typer.Option(None, "--project"),
        all_modules: bool = typer.Option(False, "--all"),
        timeout: float = typer.Option(300, "--timeout", min=1, max=3600),
    ) -> None:
        """Build the preview target and list its @Preview functions."""
        call(ctx, "preview_list", reference=reference, project_dir=project,
             include_dependencies=all_modules, timeout_s=timeout)

    @previews.command("render")
    def preview_render(
        ctx: typer.Context, reference: str | None = typer.Argument(None),
        fqn: list[str] = typer.Option(..., "--fqn", help="Preview name from preview list; repeatable."),
        project: str | None = typer.Option(None, "--project"),
        output: str | None = typer.Option(None, "--output", "-o"),
        component_tree: bool = typer.Option(True, "--component-tree/--no-component-tree"),
        daemon: bool = typer.Option(False, "--daemon", help="Keep the preview renderer warm."),
        no_image: bool = typer.Option(False, "--no-image", help="Skip PNG output; requires component-tree export."),
        timeout: float = typer.Option(300, "--timeout", min=1, max=3600),
    ) -> None:
        """Compile and render PNGs plus semantic component trees, with no emulator."""
        call(ctx, "preview_render", reference=reference, project_dir=project, fqns=fqn,
             output=output, component_tree=component_tree, daemon=daemon, no_image=no_image,
             timeout_s=timeout)

    @previews.command("inspect")
    def preview_inspect(
        ctx: typer.Context, path: str,
        max_nodes: int = typer.Option(100, "--max-nodes", min=1, max=1000),
    ) -> None:
        """Read rendered component evidence, including semantics and source locations."""
        call(ctx, "preview_inspect", path=path, max_nodes=max_nodes)

    @previews.command("check")
    def preview_check(
        ctx: typer.Context, path: str,
        checks: str = typer.Option(..., "--checks", help="JSON array of selector/property/expected checks against preview inspect fields."),
    ) -> None:
        """Check a saved tree: exists, count, within_canvas, min_width_dp, min_height_dp, text, role."""
        run(ctx, lambda engine, fmt: emit_result(route(
            engine, "preview_check", path=path, checks=_json(checks, list),
        ), fmt, "preview_check"))

    @profile.command("start")
    def profile_start(
        ctx: typer.Context, package: str | None = typer.Option(None, "--package"),
        output: str | None = typer.Option(None, "--output", "-o"),
        preset: str | None = typer.Option(None, "--preset"),
        duration: float | None = typer.Option(None, "--duration", min=0.1),
        launch: bool = typer.Option(False, "--launch"),
        wait: bool = typer.Option(False, "--wait"),
        timeout: float = typer.Option(120, "--timeout", min=1, max=3600),
    ) -> None:
        """Start a trace; use normal AUA actions, then profile stop to save it."""
        opts: dict[str, Any] = {"launch": launch, "wait": wait}
        opts.update({k: v for k, v in {
            "package": package, "output": output, "preset": preset, "duration": duration,
        }.items() if v is not None})
        call(ctx, "development_tool", operation="profiler.systemtrace.start", options=opts,
             timeout_s=timeout)

    @profile.command("stop")
    def profile_stop(
        ctx: typer.Context, keep_session: bool = typer.Option(False, "--keep-session"),
        timeout: float = typer.Option(120, "--timeout", min=1, max=3600),
    ) -> None:
        """Stop the caller's trace, save its artifact and restore the device."""
        call(ctx, "development_tool", operation="profiler.systemtrace.stop",
             options={"keep-session": keep_session}, timeout_s=timeout)

    @profile.command("status")
    def profile_status(ctx: typer.Context, heap: bool = typer.Option(False, "--heap")) -> None:
        call(ctx, "development_tool", operation="profiler.session.status",
             profile_kind="heap" if heap else "trace")

    @profile.command("cancel")
    def profile_cancel(ctx: typer.Context, heap: bool = typer.Option(False, "--heap")) -> None:
        """Terminate only this caller's profiling session and restore its settings."""
        call(ctx, "development_tool", operation="profiler.session.terminate",
             profile_kind="heap" if heap else "trace")

    @profile.command("heap")
    def profile_heap(
        ctx: typer.Context, process: str = typer.Argument(...),
        output: str | None = typer.Option(None, "--output", "-o"),
        gc: bool = typer.Option(False, "--gc"),
        timeout: int = typer.Option(120, "--timeout", min=1, max=3570),
    ) -> None:
        """Capture a JVM heap snapshot from the leased target (API 29+)."""
        opts: dict[str, Any] = {"process": process, "gc": gc, "timeout": timeout}
        if output:
            opts["output"] = output
        call(ctx, "development_tool", operation="profiler.heapdump.capture", options=opts,
             timeout_s=timeout + 30)

    @journeys.command("validate")
    def journey_validate(ctx: typer.Context, file: str) -> None:
        """Validate authored YAML and assertion grammar without touching a device."""
        call(ctx, "journey_validate", file=file)

    @journeys.command("start")
    def journey_start(
        ctx: typer.Context, file: str,
        app_id: str | None = typer.Option(None, "--app"),
        apk: str | None = typer.Option(None, "--apk"),
        artifacts_dir: str | None = typer.Option(None, "--artifacts-dir", help="Retain the session evidence bundle here."),
        provision: bool = typer.Option(True, "--provision-target/--no-provision-target", help="Allow AUA to provision a free target when needed."),
        headed: bool = typer.Option(False, "--headed", help="Show a target started by this journey."),
    ) -> None:
        """Start the authored journey; the calling agent performs each instruction."""
        call(ctx, "journey_start", file=file, package=app_id, apk=apk,
             artifacts_dir=artifacts_dir, start_emulator=provision, headed=headed)

    @journeys.command("status")
    def journey_status(
        ctx: typer.Context, session_id: str | None = typer.Option(None, "--session-id"),
    ) -> None:
        """Show the active step, proof criteria, attempts, and remaining progress."""
        call(ctx, "journey_status", session_id=session_id)

    @journeys.command("check")
    def journey_check(
        ctx: typer.Context, session_id: str | None = typer.Option(None, "--session-id"),
    ) -> None:
        """Check the current step's assertions on one fresh frame."""
        call(ctx, "journey_check", session_id=session_id)

    @journeys.command("record")
    def journey_record(
        ctx: typer.Context,
        evidence: str = typer.Option(..., "--evidence"),
        fingerprint: str = typer.Option(..., "--fingerprint"),
        passed: bool = typer.Option(True, "--passed/--failed"),
        session_id: str | None = typer.Option(None, "--session-id"),
    ) -> None:
        """Record an agent's visual assessment, explicitly separate from machine proof."""
        call(ctx, "journey_record", evidence=evidence, fingerprint=fingerprint, passed=passed,
             session_id=session_id)

    @journeys.command("finish")
    def journey_finish(
        ctx: typer.Context,
        allow_incomplete: bool = typer.Option(False, "--allow-incomplete"),
        stop_started_target: bool = typer.Option(False, "--stop-started-target"),
        session_id: str | None = typer.Option(None, "--session-id"),
    ) -> None:
        """Report proof and clean up; incomplete work needs explicit abandonment."""
        call(ctx, "journey_finish", allow_incomplete=allow_incomplete,
             retain_started_target=not stop_started_target, session_id=session_id)
