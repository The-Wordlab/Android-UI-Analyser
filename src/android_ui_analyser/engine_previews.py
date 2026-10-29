"""Host-rendered component previews and assertions over their recorded component trees.

Preview trees are build artifacts, never live device observations or reusable tap handles.
The schema below was inspected from Android CLI 1.1.16717390's component-tree output.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import UsageError

if TYPE_CHECKING:
    from .engine import Engine

_MAX_TREE_BYTES = 16 * 1024 * 1024
_MAX_COMPONENTS = 20_000
_SCOPE = {
    "scope": "rendered_preview",
    "runtime_verified": False,
    "interactive": False,
    "note": "Preview code is compiled and rendered on the host. Device behavior, navigation, "
    "networking and touch delivery are not tested by this artifact.",
}


def preview_list(
    self: Engine,
    reference: str | None = None,
    *,
    project_dir: str | None = None,
    include_dependencies: bool = False,
    timeout_s: float = 300,
) -> dict[str, Any]:
    result = self.development_tool(
        "preview.list", arguments=[reference] if reference else [],
        options={"format": "json", "all": include_dependencies},
        project_dir=project_dir, timeout_s=timeout_s,
    )
    output = {**result, **_SCOPE}
    data = result.get("data")
    if isinstance(data, dict) and isinstance(data.get("previews"), list):
        output["previews"] = data["previews"]
        output.pop("data", None)
        output.pop("stdout", None)
        output["ok"] = bool(result.get("ok") and data.get("success"))
    return output


def preview_render(
    self: Engine,
    reference: str | None = None,
    *,
    project_dir: str | None = None,
    fqns: list[str],
    output: str | None = None,
    component_tree: bool = True,
    daemon: bool = False,
    no_image: bool = False,
    timeout_s: float = 300,
) -> dict[str, Any]:
    if not fqns or any(not isinstance(value, str) or not value.strip() for value in fqns):
        raise UsageError("preview render needs at least one FQN from preview list")
    if no_image and not component_tree:
        raise UsageError("no_image requires component_tree")
    if output:
        destination = Path(output).expanduser()
        if not destination.is_absolute():
            destination = Path(project_dir or Path.cwd()).expanduser() / destination
    else:
        destination = Path(self.config.cache.dir).expanduser() / "previews" / uuid.uuid4().hex
    result = self.development_tool(
        "preview.render", arguments=[reference] if reference else [],
        options={
            "format": "json", "fqn": fqns, "component-tree": component_tree,
            "daemon": daemon, "no-image": no_image, "output": str(destination.resolve()),
        },
        project_dir=project_dir, timeout_s=timeout_s,
    )
    output_result = {**result, **_SCOPE}
    data = result.get("data")
    if isinstance(data, dict) and data.get("success") is False:
        output_result["ok"] = False
    previews = []
    if isinstance(data, dict):
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            preview = dict(item)
            artifacts = {}
            required = (["componentTreePath"] if component_tree else []) + ([] if no_image else ["imagePath"])
            if any(not isinstance(item.get(field), str) for field in required):
                output_result["ok"] = False
            for field in ("imagePath", "componentTreePath"):
                raw = item.get(field)
                if isinstance(raw, str):
                    path = Path(raw).expanduser()
                    artifacts[field] = {"path": str(path), "exists": path.is_file()}
                    if not path.is_file():
                        output_result["ok"] = False
            preview["artifacts"] = artifacts
            previews.append(preview)
    if isinstance(data, dict):
        output_result.pop("stdout", None)
        output_result.pop("data", None)
    if not previews:
        output_result["ok"] = False
        output_result["code"] = "preview_outputs_missing"
    output_result["previews"] = previews
    return output_result


def _tree(path: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    source = Path(path).expanduser().resolve()
    try:
        if source.stat().st_size > _MAX_TREE_BYTES:
            raise UsageError("preview component tree exceeds 16 MiB")
        raw = source.read_bytes()
        tree = json.loads(raw)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise UsageError(f"cannot read preview component tree: {exc}") from exc
    if not isinstance(tree, dict) or not isinstance(tree.get("components"), list):
        raise UsageError("expected a Compose component-tree JSON with a components list")
    for dimension in ("width", "height", "densityDpi"):
        value = tree.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise UsageError(f"preview component tree needs a positive numeric {dimension}")
    nodes: list[dict[str, Any]] = []
    pending = [(value, f"component:{index}", 0) for index, value in reversed(list(enumerate(tree["components"])))]
    while pending:
        node, node_id, depth = pending.pop()
        if not isinstance(node, dict) or depth > 128 or len(nodes) >= _MAX_COMPONENTS:
            raise UsageError("invalid or excessively deep/large preview component tree")
        bounds = node.get("bounds")
        if bounds is not None and (not isinstance(bounds, dict) or any(
            isinstance(bounds.get(key), bool) or not isinstance(bounds.get(key), (int, float))
            or not math.isfinite(bounds[key]) for key in ("left", "top", "right", "bottom")
        )):
            raise UsageError(f"invalid bounds at {node_id}")
        semantics = node.get("semantics", {})
        if not isinstance(semantics, dict):
            raise UsageError(f"invalid semantics at {node_id}")
        normalized = {
            "id": node_id,
            "name": node.get("name"),
            "bounds": bounds,
            "text": semantics.get("text"),
            "role": semantics.get("role"),
            "test_tag": semantics.get("testTag"),
            "semantics": semantics,
            "source": node.get("sourceLocation"),
            "parameters": node.get("parameters"),
            "modifiers": node.get("modifiers"),
        }
        if bounds is not None:
            scale = 160 / tree["densityDpi"]
            normalized["width_dp"] = round((bounds["right"] - bounds["left"]) * scale, 3)
            normalized["height_dp"] = round((bounds["bottom"] - bounds["top"]) * scale, 3)
        nodes.append(normalized)
        children = node.get("children", [])
        if not isinstance(children, list):
            raise UsageError(f"invalid children at {node_id}")
        pending.extend((child, f"{node_id}/{index}", depth + 1) for index, child in reversed(list(enumerate(children))))
    metadata = {key: value for key, value in tree.items() if key != "components"}
    metadata["path"] = str(source)
    return metadata, nodes, hashlib.sha256(raw).hexdigest()


def preview_inspect(self: Engine, path: str, *, max_nodes: int = 100) -> dict[str, Any]:
    if isinstance(max_nodes, bool) or not 1 <= max_nodes <= 1000:
        raise UsageError("max_nodes must be between 1 and 1000")
    metadata, nodes, digest = _tree(path)
    return {
        "ok": True, **_SCOPE,
        "preview": metadata,
        "sha256": digest,
        "components": nodes[:max_nodes],
        "component_count": len(nodes),
        "truncated": len(nodes) > max_nodes,
        "id_scope": "this component-tree artifact only; these IDs cannot be tapped",
    }


def _select(nodes: list[dict[str, Any]], selector: Any) -> list[dict[str, Any]]:
    if not isinstance(selector, dict) or not selector:
        raise UsageError("preview check selector must be a nonempty mapping")
    unknown = set(selector) - {"id", "name", "text", "role", "test_tag"}
    if unknown or any(not isinstance(value, str) for value in selector.values()):
        raise UsageError("preview selectors accept string id/name/text/role/test_tag fields only")
    return [node for node in nodes if all(node.get(key) == value for key, value in selector.items())]


def preview_check(self: Engine, path: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Check explicit static artifact facts, without inferring runtime/accessibility behavior."""
    if not isinstance(checks, list) or not checks or len(checks) > 100:
        raise UsageError("provide between 1 and 100 preview checks")
    metadata, nodes, digest = _tree(path)
    results = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) - {"selector", "property", "expected"}:
            raise UsageError("each preview check needs selector, property and expected")
        prop = check.get("property")
        if prop not in {"exists", "count", "within_canvas", "min_width_dp", "min_height_dp", "text", "role"}:
            raise UsageError(f"unsupported preview check property {prop!r}")
        if "expected" not in check:
            raise UsageError("preview check needs expected")
        expected = check["expected"]
        selected = _select(nodes, check.get("selector"))
        if prop in {"exists", "within_canvas"} and not isinstance(expected, bool):
            raise UsageError(f"{prop} expected must be boolean")
        if prop in {"count", "min_width_dp", "min_height_dp"} and (
            isinstance(expected, bool) or not isinstance(expected, (int, float))
            or not math.isfinite(expected) or expected < 0
        ):
            raise UsageError(f"{prop} expected must be a nonnegative number")
        if prop in {"text", "role"} and not isinstance(expected, str):
            raise UsageError(f"{prop} expected must be a string")
        if prop == "exists":
            actual: Any = bool(selected)
            ok = actual == expected
        elif prop == "count":
            actual = len(selected)
            ok = actual == expected
        elif len(selected) != 1:
            actual = {"matches": len(selected)}
            ok = False
        elif prop == "within_canvas":
            bounds = selected[0]["bounds"]
            if bounds is None:
                actual = None
                ok = False
            else:
                actual = (
                    0 <= bounds["left"] < bounds["right"] <= metadata["width"]
                    and 0 <= bounds["top"] < bounds["bottom"] <= metadata["height"]
                )
                ok = actual == expected
        elif prop.startswith("min_"):
            actual = selected[0].get(prop.removeprefix("min_"))
            ok = actual is not None and actual >= expected
        else:
            actual = selected[0].get(prop)
            ok = actual == expected
        results.append({
            "index": index, "ok": ok, "property": prop, "expected": expected,
            "actual": actual, "matched_ids": [node["id"] for node in selected],
        })
    return {
        "ok": all(result["ok"] for result in results), **_SCOPE,
        "verified": all(result["ok"] for result in results),
        "proof_kind": "static_component_tree_assertions",
        "path": metadata["path"], "sha256": digest, "checks": results,
    }
