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
    if kind == "array":
        items = _type_of(spec, resolved.get("items"), depth + 1) if depth < _MAX_DEPTH else "any"
        return f"array<{items}>"
    if kind:
        return str(kind)
    if ref or "properties" in resolved or "allOf" in resolved:
        return "object"
    return "any"


def _enum_of(spec: dict[str, Any], node: Any, depth: int = 0) -> list[str] | None:
    """The values a field may hold, or ``None`` when it is not restricted to a set."""

    resolved, _ = _deref(spec, node)
    if "enum" in resolved:
        return sorted(str(v) for v in resolved["enum"] if v is not None)
    if "const" in resolved:
        return [str(resolved["const"])]
    variants = [v for v in _variants(resolved) if not _is_null(spec, v)]
    if variants and depth < _MAX_DEPTH:
        sets = [_enum_of(spec, v, depth + 1) for v in variants]
        return None if any(s is None for s in sets) else sorted(set().union(*sets))  # type: ignore[arg-type]
    if resolved.get("type") == "array" and depth < _MAX_DEPTH:
        return _enum_of(spec, resolved.get("items"), depth + 1)
    return None


def _alternatives(kind: str) -> set[str]:
    """``integer|array<a|b>`` → ``{"integer", "array<a|b>"}``."""

    found, depth, start = set(), 0, 0
    for i, char in enumerate(kind):
        depth += {"<": 1, ">": -1}.get(char, 0)
        if char == "|" and depth == 0:
            found.add(kind[start:i])
            start = i + 1
    found.add(kind[start:])
    return found


def _merge(out: dict[str, dict[str, Any]], found: dict[str, dict[str, Any]]) -> None:
    """Fold *found* into *out*, as alternatives: a union's branches must not overwrite each other."""

    for path, new in found.items():
        old = out.get(path)
        if old is None:
            out[path] = new
            continue
        enums = (
            None
            if old["enum"] is None or new["enum"] is None
            else set(old["enum"]) | set(new["enum"])
        )
        out[path] = {
            **old,
            "type": "|".join(sorted(_alternatives(old["type"]) | _alternatives(new["type"]))),
            "required": old["required"] and new["required"],
            "nullable": old["nullable"] or new["nullable"],
            "enum": None if enums is None else sorted(enums),
        }


def fields(
    spec: dict[str, Any],
    node: Any,
    prefix: str = "",
    stack: tuple[str, ...] = (),
    owner: str | None = None,
    local: str = "",
) -> dict[str, dict[str, Any]]:
    """Every field under *node* as ``a.b[].c`` → type, required, nullable, enum.

    Each field also names the schema component that declares it (``schema``) and its path inside
    that component (``schema_field``), so one change to a shared model reads as one change.
    """

    out: dict[str, dict[str, Any]] = {}
    resolved, ref = _deref(spec, node)
    if ref:
        if ref in stack:
            return out
        stack = (*stack, ref)
        owner, local = ref.rsplit("/", 1)[-1], ""
    if len(stack) > _MAX_DEPTH:
        return out
    for variant in _variants(resolved):
        if not _is_null(spec, variant):
            _merge(out, fields(spec, variant, prefix, stack, owner, local))
    for part in resolved.get("allOf") or []:
        out.update(fields(spec, part, prefix, stack, owner, local))
    if resolved.get("type") == "array" or "items" in resolved:
        items = resolved.get("items") or {}
        _merge(out, fields(spec, items, prefix + "[]", stack, owner, local + "[]"))
    required = set(resolved.get("required") or [])
    for name, sub in (resolved.get("properties") or {}).items():
        path = f"{prefix}.{name}" if prefix else name
        here = f"{local}.{name}" if local else name
        out[path] = {
            "type": _type_of(spec, sub),
            "required": name in required,
            "nullable": _nullable(spec, sub),
            "enum": _enum_of(spec, sub),
            "schema": owner,
            "schema_field": here,
        }
        _merge(out, fields(spec, sub, path, stack, owner, here))
    return out


def _json_schema(content: Any) -> Any:
    if not isinstance(content, dict) or not content:
        return None
    media = content.get("application/json") or next(iter(content.values()))
    return media.get("schema") if isinstance(media, dict) else None


def _success_bodies(spec: dict[str, Any], op: dict[str, Any]) -> list[Any]:
    """The schema of every 2xx response that has a body (``{}`` when it is undescribed)."""

    bodies = []
    for status, raw in (op.get("responses") or {}).items():
        if str(status).startswith("2"):
            response, _ = _deref(spec, raw)
            if response.get("content"):
                bodies.append(_json_schema(response["content"]) or {})
    return bodies


def _response_fields(spec: dict[str, Any], op: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for schema in _success_bodies(spec, op):
        _merge(out, fields(spec, schema))
    return out


def untyped_response(spec: dict[str, Any], op: dict[str, Any]) -> bool:
    """A success body the spec does not describe, so only removing the operation is visible."""

    return any(
        not fields(spec, schema) and _type_of(spec, schema) in ("any", "object")
        for schema in _success_bodies(spec, op)
    )


def _params(spec: dict[str, Any], params: list[Any]) -> dict[str, dict[str, Any]]:
    found = {}
    for raw in params:
        param, _ = _deref(spec, raw)
        schema = param.get("schema") or {}
        found[f"{param.get('in')}:{param.get('name')}"] = {
            # A path parameter is part of the route the client already builds, not a new demand.
            "demanded": bool(param.get("required")) and param.get("in") != "path",
            "type": _type_of(spec, schema),
            "enum": _enum_of(spec, schema),
        }
    return found


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


def _comparable(old: str, new: str) -> bool:
    return "any" not in old and "any" not in new


def _widens(old: str, new: str) -> bool:
    """A response may now send a type a client built against *old* has never seen."""

    return _comparable(old, new) and bool(_alternatives(new) - _alternatives(old))


def _narrows(old: str, new: str) -> bool:
    """A request may no longer accept a type a client built against *old* sends."""

    return _comparable(old, new) and bool(_alternatives(old) - _alternatives(new))


def _quoted(values: list[str]) -> str:
    return ", ".join(f"`{v}`" for v in values)


def _adds(old: list[str] | None, new: list[str] | None) -> str | None:
    if old is None:
        return None
    if new is None:
        return "no longer an enum"
    extra = sorted(set(new) - set(old))
    return f"adds {_quoted(extra)}" if extra else None


def _rejects(old: list[str] | None, new: list[str] | None) -> str | None:
    if new is None:
        return None
    if old is None:
        return f"now only accepts {_quoted(new)}"
    gone = sorted(set(old) - set(new))
    return f"rejects {_quoted(gone)}" if gone else None


def _at(path: str, record: dict[str, Any]) -> dict[str, Any]:
    where: dict[str, Any] = {"field": path}
    if record.get("schema"):
        where |= {"schema": record["schema"], "schema_field": record["schema_field"]}
    return where


def _response_changes(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    removed: set[str] = set()
    respelled = {_spelling(path): path for path in after}
    for path, old in before.items():
        new = after.get(path)
        where = _at(path, old)
        if new is None:
            # One finding for a removed object, not one per field inside it.
            if not _under_reported(path, removed):
                change = {"change": "response_field_removed", **where}
                # `created_at` → `createdAt` is still a removal for a client decoding the old
                # key, but the agent needs to know to look for the old spelling.
                if _spelling(path) in respelled:
                    leaf = respelled[_spelling(path)].rsplit(".", 1)[-1]
                    change["detail"] = f"renamed to `{leaf}`?"
                changes.append(change)
            removed.add(path)
            continue
        if _widens(old["type"], new["type"]):
            detail = f"{old['type']} → {new['type']}"
            changes.append({"change": "response_field_type_changed", **where, "detail": detail})
        if old["required"] and not new["required"]:
            changes.append({"change": "response_field_now_optional", **where})
        if not old["nullable"] and new["nullable"]:
            changes.append({"change": "response_field_now_nullable", **where})
        added = _adds(old["enum"], new["enum"])
        if added:
            # An addition, and still a break: a strict enum fails on a value it has never seen.
            changes.append({"change": "response_enum_value_added", **where, "detail": added})
    return changes


def _request_changes(
    spec_before: dict[str, Any],
    spec_after: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    old_params = _params(spec_before, before["parameters"])
    for key, new in sorted(_params(spec_after, after["parameters"]).items()):
        old = old_params.get(key)
        if new["demanded"] and (old is None or not old["demanded"]):
            changes.append({"change": "request_parameter_now_required", "field": key})
        if old is None:
            continue
        if _narrows(old["type"], new["type"]):
            detail = f"{old['type']} → {new['type']}"
            changes.append(
                {"change": "request_parameter_type_changed", "field": key, "detail": detail}
            )
        rejected = _rejects(old["enum"], new["enum"])
        if rejected:
            changes.append(
                {"change": "request_enum_value_removed", "field": key, "detail": rejected}
            )
    old_required, old_body = _body(spec_before, before["op"])
    new_required, new_body = _body(spec_after, after["op"])
    if new_required and not old_required:
        changes.append({"change": "request_body_now_required"})
    for path, new in new_body.items():
        old = old_body.get(path)
        where = _at(path, new)
        # An old client can only send fields that existed when it was built, so a required
        # field inside an object that is itself new asks nothing of it.
        reachable = all(parent in old_body for parent in _ancestors(path))
        if new["required"] and (old is None or not old["required"]) and reachable:
            changes.append({"change": "request_field_now_required", **where})
        if old is None:
            continue
        if _narrows(old["type"], new["type"]):
            detail = f"{old['type']} → {new['type']}"
            changes.append({"change": "request_field_type_changed", **where, "detail": detail})
        if old["nullable"] and not new["nullable"]:
            changes.append({"change": "request_field_no_longer_nullable", **where})
        rejected = _rejects(old["enum"], new["enum"])
        if rejected:
            changes.append({"change": "request_enum_value_removed", **where, "detail": rejected})
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
