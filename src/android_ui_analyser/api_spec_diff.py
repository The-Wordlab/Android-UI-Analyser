"""Diff two OpenAPI specs into the changes an already-shipped client can feel.

Only changes that can break a client built against the *base* spec are reported: an operation
that disappeared, a response field that disappeared, changed type or may now be missing, and a
request parameter or body field the server now requires. Additions are counted, never reported —
an old client does not know they exist.

Paths are compared after :func:`normalize_path`, which is also what the client scanners use, so
``/items/{item_id}`` in a spec and ``items/\\(id)`` in Swift or ``items/${id}`` in JavaScript
meet as ``/items/{}``.
"""

from __future__ import annotations

import re
from typing import Any

_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")

# `{id}` (OpenAPI, Retrofit, Python f-strings), `${id}` (JS templates), `\(id)` (Swift).
_PARAM = re.compile(r"\$\{[^}]*\}|\\\([^)]*\)|\{[^}]*\}")

# Deep enough for any real response model; a guard against pathological specs, not a limit
# anyone should meet.
_MAX_DEPTH = 12


def normalize_path(raw: str, base_path: str = "") -> str:
    """One spelling for one route, whichever codebase wrote it.

    A relative path is joined to *base_path* (Retrofit's base-URL rule: a leading ``/`` means
    host root). Query, fragment and scheme/host are dropped, and every path parameter becomes
    ``{}`` so a renamed placeholder is not a different route.
    """

    path = raw.strip().split("#", 1)[0].split("?", 1)[0]
    if "://" in path:
        rest = path.split("://", 1)[1]
        path = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
    if not path.startswith("/"):
        path = base_path.rstrip("/") + "/" + path if base_path else "/" + path
    path = _PARAM.sub("{}", path)
    path = re.sub(r"/{2,}", "/", path)
    return path.rstrip("/") or "/"


def operation_key(method: str, path: str) -> str:
    return f"{method.upper()} {path}"


def operations(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``"GET /a/{}"`` → the operation plus the path-level parameters it inherits."""

    found: dict[str, dict[str, Any]] = {}
    for raw_path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        for method in _METHODS:
            op = item.get(method)
            if isinstance(op, dict):
                found[operation_key(method, normalize_path(raw_path))] = {
                    "path": raw_path,
                    "op": op,
                    "parameters": [*(item.get("parameters") or []), *(op.get("parameters") or [])],
                }
    return found


def _deref(spec: dict[str, Any], node: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(node, dict):
        return {}, None
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return node, None
    target: Any = spec
    for part in ref[2:].split("/"):
        target = (
            target.get(part.replace("~1", "/").replace("~0", "~"), {})
            if isinstance(target, dict)
            else {}
        )
    return (target if isinstance(target, dict) else {}), ref


def _variants(node: dict[str, Any]) -> list[Any]:
    return list(node.get("anyOf") or node.get("oneOf") or [])


def _is_null(spec: dict[str, Any], node: Any) -> bool:
    resolved, _ = _deref(spec, node)
    return resolved.get("type") == "null"


def _nullable(spec: dict[str, Any], node: Any) -> bool:
    resolved, _ = _deref(spec, node)
    kind = resolved.get("type")
    return bool(
        resolved.get("nullable")
        or (isinstance(kind, list) and "null" in kind)
        or any(_is_null(spec, v) for v in _variants(resolved))
    )


def _type_of(spec: dict[str, Any], node: Any, depth: int = 0) -> str:
    resolved, ref = _deref(spec, node)
    variants = [v for v in _variants(resolved) if not _is_null(spec, v)]
    if variants and depth < _MAX_DEPTH:
        return "|".join(sorted({_type_of(spec, v, depth + 1) for v in variants}))
    kind = resolved.get("type")
    if isinstance(kind, list):
        return "|".join(sorted(k for k in kind if k != "null")) or "null"
    if kind:
        return str(kind)
    if ref or "properties" in resolved or "allOf" in resolved:
        return "object"
    return "any"


def fields(
    spec: dict[str, Any], node: Any, prefix: str = "", stack: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    """Every field under *node* as ``a.b[].c`` → type, required, nullable."""

    out: dict[str, dict[str, Any]] = {}
    resolved, ref = _deref(spec, node)
    if ref:
        if ref in stack:
            return out
        stack = (*stack, ref)
    if len(stack) > _MAX_DEPTH:
        return out
    for variant in _variants(resolved):
        if not _is_null(spec, variant):
            out.update(fields(spec, variant, prefix, stack))
    for part in resolved.get("allOf") or []:
        out.update(fields(spec, part, prefix, stack))
    if resolved.get("type") == "array" or "items" in resolved:
        out.update(fields(spec, resolved.get("items") or {}, prefix + "[]", stack))
    required = set(resolved.get("required") or [])
    for name, sub in (resolved.get("properties") or {}).items():
        path = f"{prefix}.{name}" if prefix else name
        out[path] = {
            "type": _type_of(spec, sub),
            "required": name in required,
            "nullable": _nullable(spec, sub),
        }
        out.update(fields(spec, sub, path, stack))
    return out


def _json_schema(content: Any) -> Any:
    if not isinstance(content, dict) or not content:
        return None
    media = content.get("application/json") or next(iter(content.values()))
    return media.get("schema") if isinstance(media, dict) else None


def _response_fields(spec: dict[str, Any], op: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for status in sorted((op.get("responses") or {}), key=str):
        if str(status).startswith("2"):
            response, _ = _deref(spec, op["responses"][status])
            schema = _json_schema(response.get("content"))
            return fields(spec, schema) if schema is not None else {}
    return {}


def _required_params(spec: dict[str, Any], params: list[Any]) -> set[str]:
    required = set()
    for raw in params:
        param, _ = _deref(spec, raw)
        # A path parameter is part of the route the client already builds, not a new demand.
        if param.get("required") and param.get("in") != "path":
            required.add(f"{param.get('in')}:{param.get('name')}")
    return required


def _body(spec: dict[str, Any], op: dict[str, Any]) -> tuple[bool, dict[str, dict[str, Any]]]:
    body, _ = _deref(spec, op.get("requestBody"))
    schema = _json_schema(body.get("content"))
    return bool(body.get("required")), (fields(spec, schema) if schema is not None else {})


def _ancestors(path: str) -> list[str]:
    """``a.b[].c`` → ``["a", "a.b"]``: the fields a value at *path* is nested in."""

    parts = path.split(".")
    return [".".join(parts[:i]).removesuffix("[]") for i in range(1, len(parts))]


def _under_reported(path: str, reported: set[str]) -> bool:
    return any(
        path.startswith(parent + ".") or path.startswith(parent + "[]") for parent in reported
    )


def _spelling(path: str) -> str:
    return path.replace("_", "").replace("-", "").lower()


def _response_changes(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    removed: set[str] = set()
    respelled = {_spelling(path): path for path in after}
    for path, old in before.items():
        new = after.get(path)
        if new is None:
            # One finding for a removed object, not one per field inside it.
            if not _under_reported(path, removed):
                change = {"change": "response_field_removed", "field": path}
                # `created_at` → `createdAt` is still a removal for a client decoding the old
                # key, but the agent needs to know to look for the old spelling.
                if _spelling(path) in respelled:
                    change["detail"] = f"renamed to `{respelled[_spelling(path)]}`?"
                changes.append(change)
            removed.add(path)
            continue
        if "any" not in (old["type"], new["type"]) and old["type"] != new["type"]:
            changes.append(
                {
                    "change": "response_field_type_changed",
                    "field": path,
                    "detail": f"{old['type']} → {new['type']}",
                }
            )
        was_guaranteed = old["required"] and not old["nullable"]
        if was_guaranteed and (not new["required"] or new["nullable"]):
            changes.append(
                {
                    "change": "response_field_now_optional",
                    "field": path,
                    "detail": "no longer required" if not new["required"] else "now nullable",
                }
            )
    return changes


def _request_changes(
    spec_before: dict[str, Any],
    spec_after: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    new_params = _required_params(spec_after, after["parameters"]) - _required_params(
        spec_before, before["parameters"]
    )
    for param in sorted(new_params):
        changes.append({"change": "request_parameter_now_required", "field": param})
    old_required, old_body = _body(spec_before, before["op"])
    new_required, new_body = _body(spec_after, after["op"])
    if new_required and not old_required:
        changes.append({"change": "request_body_now_required"})
    for path, new in new_body.items():
        old = old_body.get(path)
        # An old client can only send fields that existed when it was built, so a required
        # field inside an object that is itself new asks nothing of it.
        reachable = all(parent in old_body for parent in _ancestors(path))
        if new["required"] and (old is None or not old["required"]) and reachable:
            changes.append({"change": "request_field_now_required", "field": path})
        elif (
            old is not None
            and "any" not in (old["type"], new["type"])
            and old["type"] != new["type"]
        ):
            changes.append(
                {
                    "change": "request_field_type_changed",
                    "field": path,
                    "detail": f"{old['type']} → {new['type']}",
                }
            )
    return changes


def diff_specs(base: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    """Client-visible changes from *base* to *head*, keyed by operation."""

    before, after = operations(base), operations(head)
    changes: list[dict[str, Any]] = []
    for key, old in before.items():
        new = after.get(key)
        if new is None:
            changes.append({"operation": key, "change": "operation_removed"})
            continue
        for change in [
            *_response_changes(
                _response_fields(base, old["op"]), _response_fields(head, new["op"])
            ),
            *_request_changes(base, head, old, new),
        ]:
            changes.append({"operation": key, **change})
    return {
        "operations": {"base": len(before), "head": len(after)},
        "added_operations": sorted(set(after) - set(before)),
        "changes": changes,
    }


def match_operations(method: str, path: str, known: dict[str, Any]) -> list[str]:
    """The operations a client call reaches; empty when the spec serves no such route.

    A spec ``{}`` segment accepts any client segment, and a client ``{}`` accepts any spec
    segment. The closest route wins: a literal segment matching itself beats a placeholder
    matching a placeholder, which beats a placeholder standing in for a literal — so
    ``/items/featured`` reaches ``/items/featured`` rather than ``/items/{}``, as a router would,
    and a client's ``/items/{}`` reaches ``/items/{}`` rather than ``/items/featured``. A client
    whose scanner cannot see the method (``*``) reaches every method on the winning route.
    """

    wanted = path.split("/")
    best = -1
    winners: list[str] = []
    for key in known:
        op_method, op_path = key.split(" ", 1)
        if method != "*" and method != op_method:
            continue
        segments = op_path.split("/")
        if len(segments) != len(wanted):
            continue
        score = 0
        for mine, theirs in zip(wanted, segments, strict=True):
            if mine == theirs:
                score += 1 if mine == "{}" else 2
            elif mine != "{}" and theirs != "{}":
                break
        else:
            if score > best:
                best, winners = score, [key]
            elif score == best:
                winners.append(key)
    return winners
