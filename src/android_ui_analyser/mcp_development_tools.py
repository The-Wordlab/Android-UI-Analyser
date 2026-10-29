"""Discoverable schemas for optional developer tools and authored test evidence."""

from __future__ import annotations

from typing import Any

import mcp.types as types


def _tool(name: str, description: str, properties: dict[str, Any] | None = None,
          required: tuple[str, ...] = ()) -> types.Tool:
    return types.Tool(name=name, description=description, inputSchema={
        "type": "object", "properties": properties or {},
        "required": list(required), "additionalProperties": False,
    })


def tool_definitions() -> list[types.Tool]:
    text = {"type": "string"}
    project = {"type": "string", "description": "Absolute project working directory on the tool host."}
    timeout = {"type": "number", "minimum": 1, "maximum": 3600, "default": 300}
    session = {"session_id": text}
    journey_source = {
        "file": {**text, "description": "Path to an authored journey YAML file; provide file OR yaml."},
        "yaml": {**text, "description": "Inline journey YAML. Required: version: 1, name, steps. Each step has id, instruction and either assertions (AUA assertion list) or review (visual assessment question). Provide yaml OR file."},
    }
    check = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "selector": {
                "type": "object", "minProperties": 1, "additionalProperties": False,
                "properties": dict.fromkeys(("id", "name", "text", "role", "test_tag"), text),
                "description": "Exact component fields from preview_inspect; combine fields to disambiguate.",
            },
            "property": {"enum": ["exists", "count", "within_canvas", "min_width_dp", "min_height_dp", "text", "role"]},
            "expected": {"type": ["boolean", "number", "string"], "description": "Boolean for exists/within_canvas; nonnegative number for count/minimum size; string for text/role."},
        },
        "required": ["selector", "property", "expected"],
    }
    definitions = [
        _tool("development_tools_status", "Report optional developer-tool installation/version without connecting to a device."),
        _tool("development_tools_setup", "Explicitly install the selected platform's optional developer tools from the official distribution, for this user."),
        _tool("development_tools_catalogue", "Discover supported operations, accepted options, effects and cleanup. Inspect an operation before calling development_tool.", {"operation": text}),
        _tool("development_tool", "Run one catalogued optional developer operation. Host builds/previews/docs do not need a device. Device operations use AUA's leased target; target/session overrides are forbidden. Actions return an observation. Profiling has durable cleanup. Never repeat an uncertain mutation.", {
            "operation": text,
            "arguments": {"type": "array", "items": text, "default": []},
            "options": {"type": "object", "description": "Accepted option names/values from the catalogue; no leading --."},
            "project_dir": project, "timeout_s": {**timeout, "default": 120},
            "confirmed": {"type": "boolean", "default": False},
            "profile_kind": {"type": "string", "enum": ["trace", "heap"], "description": "For profiler session status/terminate, select this caller's trace or heap session; never accepts another caller's session ID."},
        }, ("operation",)),
        _tool("preview_list", "Compile a Compose preview target and discover its @Preview functions. No device or IDE required; this still performs a build.", {
            "reference": text, "project_dir": project, "include_dependencies": {"type": "boolean", "default": False},
            "timeout_s": timeout,
        }),
        _tool("preview_render", "Render selected Compose previews to PNG plus component-tree JSON with bounds, source and semantics. Reuse these artifacts with preview_inspect/check; rendered previews are not evidence of runtime navigation.", {
            "reference": text, "project_dir": project,
            "fqns": {"type": "array", "items": text, "minItems": 1},
            "output": text, "component_tree": {"type": "boolean", "default": True},
            "daemon": {"type": "boolean", "default": False},
            "no_image": {"type": "boolean", "default": False, "description": "Skip PNG output; requires component_tree=true."}, "timeout_s": timeout,
        }, ("fqns",)),
        _tool("preview_inspect", "Read a saved preview's component tree, semantics, bounds and source locations. This does not inspect a live device.", {
            "path": text, "max_nodes": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
        }, ("path",)),
        _tool("preview_check", "Evaluate supported deterministic checks on a saved preview tree; inspect it first to discover available evidence. Does not prove runtime behavior.", {
            "path": text, "checks": {"type": "array", "items": check, "minItems": 1, "maxItems": 100},
        }, ("path", "checks")),
        _tool("journey_validate", "Validate an authored journey YAML without starting a device. Each step contains natural-language instructions plus deterministic assertions or explicit visual review.", {
            **journey_source,
        }),
        _tool("journey_start", "Start an authored journey in AUA's ordinary goal session. The calling agent performs the instruction, then calls journey_check or journey_record; instructions are not automatically executed.", {
            **journey_source, "package": text, "apk": text,
            "artifacts_dir": {**text, "description": "Directory for the full session evidence bundle."},
            "start_emulator": {"type": "boolean", "default": True, "description": "Allow provisioning a free target; false requires an available target."},
            "headed": {"type": "boolean", "default": False, "description": "Show a target started for this journey."},
        }),
        _tool("journey_status", "Read the current authored step, its assertions and remaining progress.", session),
        _tool("journey_check", "Evaluate the current journey step's deterministic assertions on one fresh frame; failure never advances progress.", session),
        _tool("journey_record", "Record an agent's explicit visual assessment against the exact previously observed fingerprint. This is labeled agent-assessed, never machine-verified.", {
            **session, "evidence": text, "fingerprint": text,
            "passed": {"type": "boolean", "default": True},
        }, ("evidence", "fingerprint")),
        _tool("journey_finish", "Finish a journey with separate deterministic and agent-assessed outcomes, using the ordinary session cleanup path.", {
            **session, "allow_incomplete": {"type": "boolean", "default": False},
            "retain_started_target": {"type": "boolean", "default": True},
        }),
    ]
    for definition in definitions:
        if definition.name in {"journey_validate", "journey_start"}:
            definition.inputSchema["oneOf"] = [{"required": ["file"]}, {"required": ["yaml"]}]
    return definitions


TOOL_NAMES = frozenset(tool.name for tool in tool_definitions())
