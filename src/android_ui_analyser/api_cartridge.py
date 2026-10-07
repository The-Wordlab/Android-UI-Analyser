"""API cartridges: send a backend the requests a client app sends, with no device.

A cartridge is a small YAML file: the requests in order, the values one response hands to a
later request, the inputs a caller may change, and the secrets it needs, named but never
stored. An agent writes one from the client's networking code; ``aua api run`` sends it to any
backend. ``--save-expect`` records each response's shape at the end of the file, so a later run
against a changed backend reports a field that went missing or changed type. Values are never
compared: ids, timestamps and generated text differ on every run.
"""

from __future__ import annotations

import codecs
import json
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import yaml

from .atomic import atomic_write_text
from .errors import UsageError

FORMAT = 1

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][\w.\-]*)\s*\}\}")
_STEP_ID = re.compile(r"^[A-Za-z_][\w\-]*$")
_SAVE_NAME = re.compile(r"^[A-Za-z_]\w*$")
_EXPECT_BLOCK = re.compile(r"^expect:", re.MULTILINE)
# Response fields whose string values a run's preview hides, so a token never reaches the
# caller's log. A number under such a name is a count (`totalTokens`), never a secret.
_SECRET_KEY = re.compile(
    r"token|secret|passw|api[-_]?key|authorization|cookie|session[-_]?id", re.IGNORECASE
)
_EXPECT_HEADER = (
    "# Written by `aua api run --save-expect`: each step's response shape. Delete a line to stop\n"
    "# checking that field; keep `expect:` the last key in this file.\n"
)
_SAMPLE = "api-cartridge-sample.yaml"


def sample_text() -> str:
    """The annotated example cartridge an agent starts from."""
    return resources.files("android_ui_analyser").joinpath("data", _SAMPLE).read_text("utf-8")


# --- loading -----------------------------------------------------------------------------------


def load(path: str | Path) -> dict[str, Any]:
    """Read and check a cartridge; every mistake is a usage error that names the step."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise UsageError(f"no cartridge at {p}", hint="Start one with `aua api sample > FILE`.")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise UsageError(f"{p} is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("cartridge") != FORMAT:
        raise UsageError(
            f"{p} is not an AUA API cartridge",
            hint=f"The file must start with `cartridge: {FORMAT}`; see `aua api sample`.",
        )
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        raise UsageError(f"{p} has no steps")
    seen: set[str] = set()
    for n, step in enumerate(steps, 1):
        where = f"step {n}"
        if not isinstance(step, dict):
            raise UsageError(f"{where} is not a mapping")
        sid = step.get("id")
        if not isinstance(sid, str) or not _STEP_ID.match(sid):
            raise UsageError(f"{where} needs an `id` made of letters, digits, `_` or `-`")
        if sid in seen:
            raise UsageError(f"two steps are called `{sid}`")
        seen.add(sid)
        where = f"step `{sid}`"
        if not isinstance(step.get("method"), str):
            raise UsageError(f"{where} needs a `method`, e.g. GET or POST")
        path_value = step.get("path")
        if not isinstance(path_value, str) or not path_value.startswith("/"):
            raise UsageError(f"{where} needs a `path` that starts with `/`")
        if "json" in step and "body" in step:
            raise UsageError(f"{where} has both `json` and `body`; keep one")
        for key in ("headers", "query", "save"):
            if key in step and not isinstance(step[key], dict):
                raise UsageError(f"{where}: `{key}` must be a mapping")
        for name, source in (step.get("save") or {}).items():
            if not _SAVE_NAME.match(str(name)) or not isinstance(source, str):
                raise UsageError(
                    f"{where}: save `{name}` needs a plain name and a source such as "
                    "`body.id`, `header.Location` or `events.done.id`"
                )
    for key in ("headers", "inputs"):
        if key in doc and not isinstance(doc[key], dict):
            raise UsageError(f"top-level `{key}` must be a mapping")
    expect = doc.get("expect")
    if expect is not None and not isinstance(expect, dict):
        raise UsageError("`expect` must map step ids to response shapes")
    doc["_path"] = str(p)
    return doc


def _references(node: Any) -> set[str]:
    if isinstance(node, str):
        return {m.group(1) for m in _PLACEHOLDER.finditer(node)}
    if isinstance(node, dict):
        return set().union(*(_references(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(_references(v) for v in node)) if node else set()
    return set()


# --- templates ---------------------------------------------------------------------------------


class _Unsaved(Exception):
    """A template names a value no earlier step saved."""


@dataclass
class _Context:
    env: Mapping[str, str]
    inputs: dict[str, Any]
    saved: dict[str, Any] = field(default_factory=dict)
    fresh: dict[str, str] = field(default_factory=dict)

    def lookup(self, name: str) -> Any:
        scope, _, key = name.partition(".")
        if scope == "env" and key:
            return self.env[key]
        if scope == "input" and key:
            return self.inputs[key]
        if scope == "fresh" and key:
            return self.fresh.setdefault(key, str(uuid.uuid4()))
        if name in self.saved:
            return self.saved[name]
        raise _Unsaved(name)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _render(node: Any, ctx: _Context, *, keep_type: bool = False) -> Any:
    """Fill every ``{{…}}``. In a JSON body a lone placeholder keeps its value's type."""
    if isinstance(node, str):
        whole = _PLACEHOLDER.fullmatch(node.strip())
        if keep_type and whole:
            return ctx.lookup(whole.group(1))
        return _PLACEHOLDER.sub(lambda m: _text(ctx.lookup(m.group(1))), node)
    if isinstance(node, dict):
        return {k: _render(v, ctx, keep_type=keep_type) for k, v in node.items()}
    if isinstance(node, list):
        return [_render(v, ctx, keep_type=keep_type) for v in node]
    return node


def _render_path(path: str, ctx: _Context) -> str:
    return _PLACEHOLDER.sub(lambda m: quote(_text(ctx.lookup(m.group(1))), safe=""), path)


# --- responses ---------------------------------------------------------------------------------


def _parse_events(text: str) -> list[tuple[str, Any]]:
    """Server-sent events as ``(event, data)``; data is parsed JSON when it parses."""
    events: list[tuple[str, Any]] = []
    data: list[str] = []
    name = "message"
    for line in [*text.splitlines(), ""]:
        if not line:
            if data:
                raw = "\n".join(data)
                try:
                    value: Any = json.loads(raw)
                except ValueError:
                    value = raw
                events.append((name, value))
            name, data = "message", []
        elif line.startswith("event:"):
            name = line[6:].strip() or "message"
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    return events


def _dig(value: Any, parts: list[str]) -> Any:
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(part)
    return value


def _extract(source: str, body: Any, headers: httpx.Headers, events: list[tuple[str, Any]]) -> Any:
    kind, _, rest = source.partition(".")
    parts = rest.split(".") if rest else []
    if kind == "body":
        return _dig(body, parts)
    if kind == "header" and rest:
        if rest not in headers:
            raise KeyError(rest)
        return headers[rest]
    if kind == "events" and parts:
        for name, data in reversed(events):
            if name == parts[0]:
                try:
                    return _dig(data, parts[1:])
                except KeyError:
                    continue
        raise KeyError(rest)
    raise KeyError(source)


# --- shapes ------------------------------------------------------------------------------------


def shape(value: Any) -> Any:
    """A value's structure with every value dropped: what a client's decoder depends on."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        merged: Any = None
        for item in value:
            merged = _merge(merged, shape(item))
        return [] if merged is None else [merged]
    if isinstance(value, dict):
        return {str(k): shape(v) for k, v in value.items()}
    return "unknown"


def _merge(a: Any, b: Any) -> Any:
    if a is None or a == "null":
        return b
    if b == "null":
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(out.get(k), v)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if not a or not b:
            return a or b
        return [_merge(a[0], b[0])]
    return a


def events_shape(events: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, data in events:
        out[name] = _merge(out.get(name), shape(data))
    return out


def _describe(s: Any) -> str:
    if isinstance(s, dict):
        return "an object"
    if isinstance(s, list):
        return "an array"
    return str(s)


def compare(expected: Any, actual: Any, where: str = "body") -> list[dict[str, str]]:
    """What a client that decodes ``expected`` would notice in ``actual``.

    A missing field or a changed type is a ``break``; a field that is now null is a ``warn``,
    since one run cannot tell an optional field from a broken one. New fields are fine.
    """
    if expected in ("null", "unknown"):
        return []
    if actual == "null":
        return [{"level": "warn", "where": where, "change": f"now null, was {_describe(expected)}"}]
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{"level": "break", "where": where, "change": f"now {_describe(actual)}"}]
        found: list[dict[str, str]] = []
        for key, sub in expected.items():
            if key not in actual:
                found.append({"level": "break", "where": f"{where}.{key}", "change": "missing"})
            else:
                found += compare(sub, actual[key], f"{where}.{key}")
        return found
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [{"level": "break", "where": where, "change": f"now {_describe(actual)}"}]
        if not expected or not actual:
            return []
        return compare(expected[0], actual[0], f"{where}[]")
    if actual != expected:
        return [
            {"level": "break", "where": where, "change": f"now {_describe(actual)}, was {expected}"}
        ]
    return []


def _compare_events(expected: dict[str, Any], actual: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for name, sub in expected.items():
        if name not in actual:
            found.append(
                {"level": "warn", "where": f"events.{name}", "change": "not sent this run"}
            )
        else:
            found += compare(sub, actual[name], f"events.{name}")
    return found


# --- running -----------------------------------------------------------------------------------


class _EventClock:
    """When each kind of server-sent event first arrived, measured while the stream is read."""

    def __init__(self, started: float) -> None:
        self.started = started
        self.first: dict[str, int] = {}
        self._decode = codecs.getincrementaldecoder("utf-8")(errors="replace").decode
        self._tail = ""
        self._name = "message"
        self._data = False

    def feed(self, chunk: bytes, *, final: bool = False) -> None:
        now = round((time.perf_counter() - self.started) * 1000)
        *lines, self._tail = (self._tail + self._decode(chunk, final)).split("\n")
        if final:
            lines += [self._tail, ""]
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                if self._data:
                    self.first.setdefault(self._name, now)
                self._name, self._data = "message", False
            elif line.startswith("event:"):
                self._name = line[6:].strip() or "message"
            elif line.startswith("data:"):
                self._data = True


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {
            k: "***" if _SECRET_KEY.search(str(k)) and isinstance(v, str) else _redact(v, secrets)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "***")
    return value


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]} …({len(text) - limit} chars)… {text[-half:]}"


def _preview(body: Any, raw: str, events: list[tuple[str, Any]], secrets: list[str]) -> str:
    if events:
        lines = [f"{name}: {_text(_redact(data, secrets))}" for name, data in events]
        return "\n".join(lines)
    if body is not None:
        return json.dumps(_redact(body, secrets), ensure_ascii=False)
    return str(_redact(raw, secrets))


def run(
    path: str | Path,
    *,
    base_url: str | None = None,
    inputs: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 120.0,
    max_chars: int = 2000,
    save_expect: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Send every step in order and report what each response looked like.

    A step fails, and the run stops there, when its status is not the step's ``status`` (any
    2xx by default) or a value it must ``save`` is absent. Shape differences against the
    file's ``expect`` block are reported per step without stopping the run.
    """
    doc = load(path)
    env = os.environ if env is None else env
    base = (base_url or doc.get("base_url") or "").rstrip("/")
    if not re.match(r"^https?://", base):
        raise UsageError(
            "no backend to send the cartridge to",
            hint="Pass `--base-url https://host` or set `base_url` in the cartridge.",
        )

    declared = dict(doc.get("inputs") or {})
    for name, value in (inputs or {}).items():
        if name not in declared:
            raise UsageError(
                f"the cartridge has no input `{name}`",
                hint="It declares: " + (", ".join(sorted(declared)) or "no inputs"),
            )
        declared[name] = value

    refs = _references({"h": doc.get("headers") or {}, "s": doc["steps"]})
    env_names = sorted(
        {str(n) for n in doc.get("env") or []}
        | {r.split(".", 1)[1] for r in refs if r.startswith("env.")}
    )
    missing_env = [n for n in env_names if not env.get(n)]
    if missing_env:
        raise UsageError(
            "set these environment variables first: " + ", ".join(missing_env),
            hint="e.g. `aua config exec --env-file .env --require "
            f"{missing_env[0]} -- aua api run {doc['_path']}`",
        )
    unknown = sorted(r.split(".", 1)[1] for r in refs if r.startswith("input."))
    unknown = [n for n in unknown if n not in declared]
    if unknown:
        raise UsageError(
            "the steps use inputs the cartridge does not declare: " + ", ".join(unknown),
            hint="Add them under `inputs:` with a default value.",
        )

    ctx = _Context(env={n: env[n] for n in env_names}, inputs=declared)
    secrets = [v for v in ctx.env.values() if len(v) >= 4]
    expect_doc = doc.get("expect") or {}
    shared_headers = doc.get("headers") or {}
    results: list[dict[str, Any]] = []
    shapes: dict[str, dict[str, Any]] = {}
    failed = False

    with httpx.Client(
        transport=transport, timeout=httpx.Timeout(timeout_s, connect=10.0), follow_redirects=True
    ) as client:
        for step in doc["steps"]:
            sid = step["id"]
            row: dict[str, Any] = {
                "id": sid,
                "method": step["method"].upper(),
                "path": step["path"],
            }
            if step.get("screen"):
                row["screen"] = step["screen"]
            results.append(row)
            if failed:
                row["ok"] = False
                row["error"] = "not run: an earlier step failed"
                continue
            try:
                url = base + _render_path(step["path"], ctx)
                headers = {
                    k: _text(v)
                    for k, v in _render(
                        {**shared_headers, **(step.get("headers") or {})}, ctx
                    ).items()
                    if v is not None
                }
                query = _render(step.get("query"), ctx)
                payload: dict[str, Any] = {}
                if "json" in step:
                    payload["json"] = _render(step["json"], ctx, keep_type=True)
                elif "body" in step:
                    payload["content"] = _text(_render(step["body"], ctx)).encode()
            except _Unsaved as exc:
                row["ok"], failed = False, True
                row["error"] = f"needs `{exc.args[0]}`, which no earlier step saved"
                continue

            started = time.perf_counter()
            first_byte: float | None = None
            try:
                with client.stream(
                    row["method"], url, headers=headers, params=query, **payload
                ) as resp:
                    clock = (
                        _EventClock(started)
                        if "text/event-stream" in resp.headers.get("content-type", "")
                        else None
                    )
                    chunks = []
                    for chunk in resp.iter_bytes():
                        if first_byte is None:
                            first_byte = time.perf_counter()
                        if clock:
                            clock.feed(chunk)
                        chunks.append(chunk)
                    if clock:
                        clock.feed(b"", final=True)
                    raw = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
            except httpx.HTTPError as exc:
                row["ok"], failed = False, True
                row["error"] = f"{type(exc).__name__}: {exc}"
                continue
            row["status"] = resp.status_code
            row["ms"] = round((time.perf_counter() - started) * 1000)

            is_stream = "text/event-stream" in resp.headers.get("content-type", "")
            events = _parse_events(raw) if is_stream else []
            body: Any = None
            if not is_stream and raw.strip():
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
            if is_stream:
                row["first_byte_ms"] = round(((first_byte or time.perf_counter()) - started) * 1000)
                row["events"] = len(events)
                if clock:
                    row["event_ms"] = clock.first
            row["response"] = _clip(_preview(body, raw, events, secrets), max_chars)

            want = step.get("status")
            if (resp.status_code != want) if want is not None else not resp.is_success:
                row["ok"], failed = False, True
                row["error"] = f"status {resp.status_code}, expected {want or '2xx'}"
                continue

            saved: list[str] = []
            for name, source in (step.get("save") or {}).items():
                try:
                    ctx.saved[name] = _extract(source, body, resp.headers, events)
                    saved.append(name)
                except KeyError:
                    row["ok"], failed = False, True
                    row["error"] = f"could not save `{name}`: the response has no `{source}`"
                    break
            if failed:
                continue
            if saved:
                row["saved"] = saved

            actual: dict[str, Any] = (
                {"events": events_shape(events)} if is_stream else {"body": shape(body)}
            )
            shapes[sid] = actual
            wanted = expect_doc.get(sid) or {}
            findings: list[dict[str, str]] = []
            if "body" in wanted:
                findings += compare(wanted["body"], actual.get("body", "unknown"))
            if "events" in wanted:
                findings += _compare_events(wanted["events"], actual.get("events") or {})
            if findings:
                row["findings"] = findings
            row["ok"] = not any(f["level"] == "break" for f in findings)

    breaks = sum(f["level"] == "break" for r in results for f in r.get("findings", []))
    warns = sum(f["level"] == "warn" for r in results for f in r.get("findings", []))
    out: dict[str, Any] = {
        "ok": not failed and breaks == 0,
        "cartridge": doc["_path"],
        "name": doc.get("name"),
        "base_url": base,
        "steps": results,
        "breaks": breaks,
        "warnings": warns,
    }
    if save_expect and failed:
        out["expect"] = "not saved: a step failed, and a failed run is no baseline"
    elif save_expect:
        write_expect(doc["_path"], shapes)
        out["expect"] = f"saved {len(shapes)} response shapes to {doc['_path']}"
    elif expect_doc:
        out["expect"] = "compared with the shapes saved in the cartridge"
    else:
        out["expect"] = (
            "none saved: run once with --save-expect against a backend that works, and later "
            "runs report what changed"
        )
    return out


def write_expect(path: str | Path, shapes: dict[str, dict[str, Any]]) -> None:
    """Replace the file's trailing ``expect:`` block, keeping everything above it as written."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    match = _EXPECT_BLOCK.search(text)
    if match:
        tail = yaml.safe_load(text[match.start() :])
        if not isinstance(tail, dict) or set(tail) != {"expect"}:
            raise UsageError(
                f"`expect:` is not the last key in {p}",
                hint="Move it to the end of the file so AUA can rewrite it.",
            )
        head = text[: match.start()]
        if head.endswith(_EXPECT_HEADER):
            head = head[: -len(_EXPECT_HEADER)]
    else:
        head = text
    block = yaml.safe_dump(
        {"expect": shapes}, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    atomic_write_text(p, head.rstrip("\n") + "\n\n" + _EXPECT_HEADER + block)
