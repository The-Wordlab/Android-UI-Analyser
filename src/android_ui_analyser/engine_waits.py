"""Waiting on and checking the screen: has/expect, wait/wait_stable/wait_changed/wait_after_change/await_predicate, the locale bridge for text matching, the caller-sized wait budget, hierarchy change detection, and the background jobs that carry a long wait.

Engine methods for waits. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_waits.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import shlex
import threading
import time
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .assertions import Selector, apply_structural_filters, check_contains_all, normalize_selector
from .device import Device
from .engine_support import (
    _WAIT_FOR_FIELDS,
    _AwaitTerm,
    _is_resource_id_lookup,
    _label,
    _parse_await_terms,
    detail_tokens,
    logger,
)
from .errors import (
    JobCancelledError,
    ProviderError,
    SelectorAmbiguousError,
    StabilityTimeout,
    UsageError,
)
from .memory import AppStrings, _id_tail
from .providers.registry import run_chain
from .schema import ActionResult, AnalyzeResult, Element, HasResult, MatchMode
from .selectors import (
    _MAX_CANDIDATES,
    app_elements,
    element_digest,
    match_selector,
    nearest_elements,
    normalize_selector_prefix,
    selector_label,
)

if TYPE_CHECKING:
    from .engine import Engine


_LOCALE_CANDIDATE_CAP = 3  # translated-label retries per text miss (generic labels fan out)


_AWAIT_PREDICATE_HELD = frozenset({"satisfied", "absence-satisfied"})


def _parse_wait_for_predicate(for_: str, *, by: str, absent: bool) -> tuple[str, str, bool]:
    """Honour a leading ``!`` (and optional ``field:`` prefix) in ``wait --for``.

    ``--until``/``await-and-analyze`` already speak ``!field:value`` for "must be absent"
    (:func:`_parse_await_terms`). ``wait-and-analyze --for`` predates that grammar: it is a
    plain string plus separate ``--by``/``--absent`` flags, with no notion of ``!`` at all. An
    agent reaching for the syntax it already uses elsewhere — ``--for '!text:Loading'`` — got
    no error: the bang and the ``text:`` prefix were both swallowed into the literal search
    needle, so the wait looked for the *presence* of a string that could never appear and
    burned the full timeout even though the absence it actually asked for was already true.

    Recognise the same ``!`` convention here rather than adding a second predicate language.
    Only a leading ``!`` triggers this — a bare ``field:value`` with no bang is left as literal
    text, unchanged, so an on-screen label such as ``"Balance: $5"`` (or even ``"id: 5"``) is
    never silently reinterpreted as a selector.
    """
    if not for_.startswith("!"):
        return for_, by, absent
    remainder = for_[1:]
    prefix, sep, value = remainder.partition(":")
    field = _WAIT_FOR_FIELDS.get(prefix.strip().lower()) if sep else None
    if field is not None and value.strip():
        return value.strip(), field, True
    # No recognised field prefix (e.g. `!Loading`) — negate the literal remainder under
    # whichever `by` the caller already selected.
    return remainder, by, True


def _job_checkpoint(self: Engine) -> None:
    """Abort a supported background wait at the next safe device-read boundary."""
    event = self._current_job_cancel_event()
    if event is not None and event.is_set():
        raise JobCancelledError("background wait cancelled")


def _current_job_cancel_event(self: Engine) -> threading.Event | None:
    """Cancellation state for the job running on this thread, if any."""
    event = getattr(self._job_context, "cancel_event", None)
    if event is not None:
        return event
    # Compatibility for callers/tests that explicitly mark a single-threaded Engine as a
    # job. JobManager itself no longer writes this process-wide slot.
    return getattr(self, "_job_cancel_event", None)


def _job_sleep(self: Engine, seconds: float) -> None:
    """Sleep interruptibly when this Engine is executing a background job."""
    event = self._current_job_cancel_event()
    if event is None:
        time.sleep(seconds)
        return
    if event.wait(max(0.0, seconds)):
        raise JobCancelledError("background wait cancelled")


def _job_requires_warm_transport(self: Engine) -> None:
    raise UsageError(
        "background jobs require a warm AUA daemon or MCP server",
        hint="Enable the daemon and retry, or use the normal foreground wait command.",
    )


# These adapters are reached only when a CLI job command cannot route to the warm daemon.
# The actual job manager lives at the daemon/MCP boundary so the worker survives the short
# client process and status/cancel calls reconnect to the same Engine.
def job_start(self: Engine, **_kwargs: Any) -> None:
    self._job_requires_warm_transport()


def job_status(self: Engine, **_kwargs: Any) -> None:
    self._job_requires_warm_transport()


def job_wait(self: Engine, **_kwargs: Any) -> None:
    self._job_requires_warm_transport()


def job_cancel(self: Engine, **_kwargs: Any) -> None:
    self._job_requires_warm_transport()


def job_list(self: Engine, **_kwargs: Any) -> None:
    self._job_requires_warm_transport()


def wait_stable(
    self: Engine,
    *,
    interval_ms: int = 120,
    settle_ms: int = 200,
    timeout_ms: int = 30000,
    observe: bool = False,
    ignore_animation: bool = True,
) -> ActionResult:
    """Return once the screen stops changing for ``settle_ms`` (PRD §5, AC14).

        Cheap perceptual-hash over screenshots only — NO OCR, NO hierarchy parse. Works on
        opaque/Compose/video screens; ideal for waiting on image generation / loading.
        ``observe`` folds in a post-settle ``analyze`` — because the screen is settled, the
        returned ids are reliable (fixes the "premature observation" trap on transitions).

        When ``ignore_animation`` is True (default), per-cell grid hashing is used so that
        regions with continuous looping animation (spinners, videos, Lottie) are auto-masked
        and don't prevent settling. The screen is "settled" when all non-animated cells stop
        changing.

        ``timeout_ms`` is a request, not a guarantee: it is sized by :meth:`_bounded_wait_ms`
        like every other observation wait. A clamped wait that settles says so on its result;
        a clamped wait that expires still raises :class:`StabilityTimeout`, because "the screen
        never went quiet" is the same answer whether it was watched for 5 seconds or 60.
        """
    from . import imaging

    self._start_call()
    device = self.device
    timeout_ms, clamped_from, ceiling_ms = self._bounded_wait_ms(timeout_ms)
    deadline = time.monotonic() + timeout_ms / 1000.0
    samples = 0

    if ignore_animation:
        gs = imaging.GridSettle(streak=imaging.ANIMATION_STREAK)
        stable_since: float | None = None
        while True:
            self._job_checkpoint()
            img = device.screenshot()
            samples += 1
            now = time.monotonic()
            grid_stable = gs.feed(img)
            if grid_stable:
                if stable_since is None:
                    stable_since = now
                if (now - stable_since) * 1000.0 >= settle_ms:
                    masked = gs.masked_cells
                    detail = f"settled after {samples} samples"
                    if masked:
                        detail += f" (ignored {len(masked)} animated cells)"
                    return self._say_the_wait_was_shortened(
                        self._observe(
                            ActionResult(ok=True, action="wait-stable", detail=detail),
                            observe,
                            settle=False,  # already settled
                        ),
                        clamped_from,
                        ceiling_ms,
                    )
            else:
                stable_since = None
            if now >= deadline:
                masked = gs.masked_cells
                hint = "Increase --timeout/--settle, or the screen is still animating."
                if masked:
                    hint = (
                        f"{len(masked)} cell(s) flagged as animation and excluded; "
                        "remaining content still changing. " + hint
                    )
                # Journal before raising: a wait that burned its whole budget is the
                # single largest cost a slow run can hide, and an exception leaves the
                # normal on-the-way-out path unreached.
                self._journal_wait_gave_up(
                    "wait-stable",
                    f"screen did not settle within {timeout_ms} ms ({samples} samples)",
                )
                raise StabilityTimeout(
                    f"screen did not settle within {timeout_ms} ms ({samples} samples)",
                    hint=self._hint_for_a_shortened_wait(hint, clamped_from, ceiling_ms),
                )
            self._sleep_between_polls(interval_ms, deadline)
    else:
        last: int | None = None
        stable_since_legacy: float | None = None
        while True:
            self._job_checkpoint()
            current = imaging.dhash(device.screenshot())
            samples += 1
            now = time.monotonic()
            if last is not None and imaging.is_stable(current, last):
                if stable_since_legacy is None:
                    stable_since_legacy = now
                if (now - stable_since_legacy) * 1000.0 >= settle_ms:
                    return self._say_the_wait_was_shortened(
                        self._observe(
                            ActionResult(
                                ok=True,
                                action="wait-stable",
                                detail=f"settled after {samples} samples",
                            ),
                            observe,
                            settle=False,
                        ),
                        clamped_from,
                        ceiling_ms,
                    )
            else:
                stable_since_legacy = None
            last = current
            if now >= deadline:
                self._journal_wait_gave_up(
                    "wait-stable",
                    f"screen did not settle within {timeout_ms} ms ({samples} samples)",
                )
                raise StabilityTimeout(
                    f"screen did not settle within {timeout_ms} ms ({samples} samples)",
                    hint=self._hint_for_a_shortened_wait(
                        "Increase --timeout/--settle, or the screen is still animating.",
                        clamped_from,
                        ceiling_ms,
                    ),
                )
            self._sleep_between_polls(interval_ms, deadline)


def has(
    self: Engine,
    text: str,
    *,
    match: str = "contains",
    ignore_case: bool = False,
    ocr_fallback: bool = True,
    source: str = "auto",
    timeout_ms: int = 0,
    by: str = "text",
) -> HasResult:
    """Quick presence check — NOT the full pipeline (PRD §5, §6a T0).

        ``by="id"`` matches a resource-id (a bare tail like ``containerDetail`` too) —
        this can confirm containers the parsed element list prunes (Maestro-style
        ``assertVisible: id:``). OCR fallback only applies to text lookups.
        """
    query_field = "rid" if _is_resource_id_lookup(by) else "desc" if by == "desc" else "text"
    text = normalize_selector_prefix(query_field, text) or text
    mode = MatchMode(match)
    device = self.device
    src = (source or "auto").lower()

    # T0: hierarchy selector (short-circuits on first hit)
    clamped_from: int | None = None
    ceiling_ms = 0
    deadline: float | None = None
    if timeout_ms and timeout_ms > 0:
        timeout_ms, clamped_from, ceiling_ms = self._bounded_wait_ms(timeout_ms)
        deadline = time.monotonic() + timeout_ms / 1000.0
    if src in ("auto", "hierarchy"):
        if timeout_ms and timeout_ms > 0:
            candidates = self._locale_candidates(device, text, by)
            bounds, matched = self._wait_for_any(
                device,
                text,
                candidates,
                mode=mode,
                ignore_case=ignore_case,
                timeout_ms=timeout_ms,
                by=by,
            )
        else:
            bounds = device.find_text(text, match=mode, ignore_case=ignore_case, by=by)
            matched = None
        if bounds is not None:
            rendered = matched[0] if matched is not None else text
            return self._has_wait_result(
                HasResult(
                    found=True,
                    source="hierarchy",
                    bounds=bounds,
                    text=rendered,
                    device_locale=device.device_locale() if matched is not None else None,
                    hint=self._translated_hint(matched[0], matched[1], matched[2], text)
                    if matched is not None
                    else None,
                ),
                clamped_from,
                ceiling_ms,
            )
        translated = (
            None
            if timeout_ms and timeout_ms > 0
            else self._find_translated(device, text, mode, ignore_case, by)
        )
        if translated is not None:
            return self._has_wait_result(translated, clamped_from, ceiling_ms)
        if src == "hierarchy" or _is_resource_id_lookup(by):
            return self._has_wait_result(
                self._has_miss(device, "hierarchy", by, text), clamped_from, ceiling_ms
            )

    # T0→T3: OCR fallback (only on a hierarchy miss)
    if (src in ("auto", "vision")) and (ocr_fallback or src == "vision"):
        remaining_ms = (
            max(0, int((deadline - time.monotonic()) * 1000)) if deadline is not None else None
        )
        hit = (
            self._ocr_contains(
                device,
                text,
                mode,
                ignore_case,
                timeout_ms=remaining_ms,
            )
            if remaining_ms is None or remaining_ms > 0
            else None
        )
        if hit is not None:
            return self._has_wait_result(
                HasResult(found=True, source="ocr", bounds=hit, text=text),
                clamped_from,
                ceiling_ms,
            )

    return self._has_wait_result(
        self._has_miss(device, "hierarchy" if src != "vision" else "ocr", by, text),
        clamped_from,
        ceiling_ms,
    )


def _has_wait_result(
    self: Engine, result: HasResult, clamped_from: int | None, ceiling_ms: int
) -> HasResult:
    if clamped_from is None:
        return result
    result.wait_clamped_from_ms = clamped_from
    result.wait_ceiling_ms = ceiling_ms
    result.wait_ceiling_mode = getattr(self._job_context, "last_wait_ceiling_mode", None)
    return result


def _has_miss(self: Engine, device: Device, source: str, by: str, text: str) -> HasResult:
    return HasResult(
        found=False,
        source=source,
        device_locale=device.device_locale(),
        hint=self._text_miss_hint(device, by, text, tried_translations=True),
    )


def _find_translated(
    self: Engine, device: Device, text: str, mode: MatchMode, ignore_case: bool, by: str
) -> HasResult | None:
    """Retry a missed text lookup with the app's known renderings of the same label
        (hierarchy only — the mined strings bridge the device's UI language, §6b)."""
    for cand, loc, key in self._locale_candidates(device, text, by):
        bounds = device.find_text(cand, match=mode, ignore_case=ignore_case, by=by)
        if bounds is not None:
            return HasResult(
                found=True,
                source="hierarchy",
                bounds=bounds,
                text=cand,
                device_locale=device.device_locale(),
                hint=self._translated_hint(cand, loc, key, text),
            )
    return None


def _app_strings(self: Engine, package: str) -> AppStrings | None:
    if package not in self._strings_cache:
        mem = self._memory
        self._strings_cache[package] = mem.load_strings(package) if mem else None
    return self._strings_cache[package]


def _locale_candidates(
    self: Engine, device: Device, text: str, by: str = "text"
) -> list[tuple[str, str, str]]:
    """(label, locale, key) alternates for *text* from the app's mined strings.

        The query is matched against every locale's value of a key, so the bridge works
        in both directions (a query in the source language on a localized device and
        vice versa); the device-locale rendering ranks first, the default (source)
        value last.
        """
    from .explore import DEFAULT_LOCALE

    if _is_resource_id_lookup(by):
        return []
    wanted = text.strip().casefold()
    if not wanted:
        return []
    pkg = self._cached_package() or self.current_package()
    strings = self._app_strings(pkg) if pkg else None
    if strings is None:
        return []
    locale = device.device_locale()
    lang = locale.split("-", 1)[0].casefold() if locale else None
    out: list[tuple[str, str, str]] = []
    seen = {wanted}
    for key, per in strings.entries.items():
        if not any(v.strip().casefold() == wanted for v in per.values()):
            continue
        exact = [
            (loc, v)
            for loc, v in per.items()
            if locale and loc.casefold() == locale.casefold()
        ]
        same_language = [
            (loc, v)
            for loc, v in per.items()
            if lang
            and loc.split("-", 1)[0].casefold() == lang
            and (not locale or loc.casefold() != locale.casefold())
        ]
        ranked = [*exact, *same_language]
        if DEFAULT_LOCALE in per:
            ranked.append((DEFAULT_LOCALE, per[DEFAULT_LOCALE]))
        for loc, v in ranked:
            cand = v.strip()
            if cand.casefold() in seen:
                continue
            seen.add(cand.casefold())
            out.append((cand, loc, key))
        if len(out) >= _LOCALE_CANDIDATE_CAP:
            break
    return out[:_LOCALE_CANDIDATE_CAP]


def _translated_hint(cand: str, loc: str, key: str, original: str) -> str:
    return f"matched '{cand}' — the {loc} rendering of '{original}' (string key {key})"


def _text_miss_hint(
    self: Engine, device: Device, by: str, text: str, *, tried_translations: bool
) -> str | None:
    """Explain a text miss: the label may render translated in the device locale."""
    if _is_resource_id_lookup(by):
        return None
    candidates = self._locale_candidates(device, text, by)
    if candidates:
        cand, loc, key = candidates[0]
        tail = (
            "that rendering is not on screen either"
            if tried_translations
            else f"try '{cand}' instead"
        )
        return f"'{text}' is string key {key}, rendered {loc} as '{cand}' — {tail}"
    return self._locale_hint(by, device.device_locale())


def _locale_hint(by: str, locale: str | None) -> str | None:
    """Why a text lookup may have missed: labels render in the device locale.

        Deliberately language-neutral — the query's language is unknowable, so the hint
        fires for any known locale. Resource-id lookups are locale-proof, never hinted.
        """
    if _is_resource_id_lookup(by) or not locale:
        return None
    return (
        f"on-screen labels render in the device locale ({locale}) — a target written "
        "in another language never matches; match text observed via `analyze`, or "
        "select --by id (locale-proof)"
    )


def _wait_for_any(
    self: Engine,
    device: Device,
    text: str,
    candidates: list[tuple[str, str, str]],
    *,
    mode: MatchMode,
    ignore_case: bool,
    timeout_ms: int,
    by: str,
) -> tuple[tuple[int, int, int, int] | None, tuple[str, str, str] | None]:
    """Poll for *text* or any known translated rendering; report which one matched."""
    if not candidates:
        return (
            device.wait_for(
                text, match=mode, ignore_case=ignore_case, timeout_ms=timeout_ms, by=by
            ),
            None,
        )
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        bounds = device.find_text(text, match=mode, ignore_case=ignore_case, by=by)
        if bounds is not None:
            return bounds, None
        for cand in candidates:
            bounds = device.find_text(cand[0], match=mode, ignore_case=ignore_case, by=by)
            if bounds is not None:
                return bounds, cand
        if time.monotonic() >= deadline:
            return None, None
        self._sleep_between_polls(200.0, deadline)


def _ocr_contains(
    self: Engine,
    device: Device,
    text: str,
    mode: MatchMode,
    ignore_case: bool,
    *,
    timeout_ms: int | None = None,
) -> tuple[int, int, int, int] | None:
    if not self.factory.is_enabled("ocr"):
        return None
    chain = self.factory.build_chain("ocr")
    if not chain.providers:
        return None
    img = device.screenshot()
    provider_timeout_ms = int(self.config.timeouts.vision_ms)
    if timeout_ms is not None:
        provider_timeout_ms = min(provider_timeout_ms, max(1, timeout_ms))
    try:
        boxes, _ = run_chain(
            chain,
            lambda p: p.recognize(img),  # type: ignore[attr-defined]
            timeout_s=provider_timeout_ms / 1000.0,
        )
    except ProviderError as exc:
        logger.info("ocr fallback unavailable: %s", exc)
        return None
    import re as _re

    needle = text if not ignore_case else text.lower()
    for tb in boxes:
        hay = tb.text if not ignore_case else tb.text.lower()
        ok = False
        if mode is MatchMode.exact:
            ok = hay.strip() == needle.strip()
        elif mode is MatchMode.regex:
            flags = _re.IGNORECASE if ignore_case else 0
            ok = _re.search(text, tb.text, flags) is not None
        else:
            ok = needle in hay
        if ok:
            return tb.bounds
    return None


def _hand_back_what_is_on_screen(
    self: Engine,
    *,
    action: str,
    waited_ms: int,
    ceiling_ms: int,
    observe: bool,
    clamped_from: int | None = None,
) -> ActionResult:
    """A bounded wait that expired is a normal outcome, not a failure.

        Under a short ceiling, expiry stops meaning "this screen is broken" and starts meaning
        "not yet". Raising there would turn the common case into an error the caller has to
        catch, and would throw away the screen we just paid to read — so return it, say plainly
        that it may be mid-flight, and let the caller decide whether to ask again. One more
        function call is cheap; a blocked session is not.
        """
    result = ActionResult(
        ok=True,
        action=action,
        detail=(
            f"still moving after {waited_ms}ms (ceiling {ceiling_ms}ms) — returning the "
            "screen as it stands"
        ),
        settled_unmet=True,
    )
    result.note = (
        "the screen had not finished changing when the ceiling was reached. This is not a "
        "failure and not proof the screen is wrong: call again to see the next state, or "
        "use `--until '<predicate>'` to wait on evidence instead of on a timer."
    )
    return self._say_the_wait_was_shortened(
        self._observe(result, observe, settle=False), clamped_from, ceiling_ms
    )


def _screen_already_answers(self: Engine, *, quiet_ms: int = 120) -> bool:
    """True when the screen is holding still and has something on it.

        The question a caller means by "wait for a change" is "let me see the result". When the
        result is already up, waiting for a *further* change answers a different question and,
        on a screen with any periodic redraw, can block until the deadline. Two cheap
        hierarchy samples a short interval apart settle it without a screenshot.
        """
    # Only meaningful when the caller already holds an observation. "The change may have
    # happened while you were composing" presupposes a before-picture; with no prior
    # analyze there is nothing that could have been missed, and probing anyway would both
    # cost a read and consume the very transition the caller asked to be shown.
    cached = self._last_analyze_result
    if cached is None or not cached.elements:
        return False
    try:
        first = self.hierarchy_fingerprint()
        if not first:
            return False
        self._job_sleep(max(0.02, quiet_ms / 1000.0))
        if self.hierarchy_fingerprint() != first:
            return False  # still moving; the caller's wait is the right instrument
    except Exception:
        return False
    return True


def _wait_ceiling(self: Engine) -> tuple[int, str]:
    """The effective ceiling and its mode. The cap is read here and nowhere else.

        `perf.max_wait_ms` is the hard maximum and this is its single reader; `wait_ceiling_ms`
        is handed that number and can only return something at or below it. Keeping the read in
        one place is what `test_the_wait_ceiling_has_no_holes` pins, and the reason is the same
        as it was then: a wait that reads the ceiling itself is a wait that can size its own
        budget.
        """
    from .perf import wait_ceiling_ms

    return wait_ceiling_ms(
        int(self.config.perf.max_wait_ms), self.config, self._caller_profile()
    )


def _bounded_wait_ms(self: Engine, requested_ms: int | None) -> tuple[int, int | None, int]:
    """Bound one observation wait to the ceiling, which is at most ``perf.max_wait_ms``.

        The ceiling adapts *downwards* within that maximum, from what this caller has been
        measured to cost between calls (see :meth:`_wait_ceiling` and ``caller_latency``): a
        shell script whose re-call costs ~3.9s of tool time and no thinking has no use for a 5s
        wait, while an LLM caller that thinks for 6-39s is already at the maximum and stays
        there. Nothing in that path can raise the number — the maximum is a standing decision,
        and an agent that needs longer is expected to make another call, not hold one long wait.

        Returns ``(effective_ms, clamped_from_or_None, ceiling_ms)``. This is the ONE gate:
        every agent-facing wait sizes its deadline here rather than from the caller's
        ``timeout_ms``, so the ceiling is a property of the session and not something a
        ``--timeout`` flag can lift. Provisioning budgets do not come through here at all —
        installing an APK or booting an emulator is not an observation and legitimately takes
        minutes.

        One exemption, and it is about who is blocked rather than about how long. While this
        Engine is executing a background job the caller already holds a job id and polls
        ``job status``, so no session is stalled; clamping there would cut short the very
        long wait the `job` vocabulary exists to hold, and its own defaults (30s wait-stable,
        60s await) would all collapse to the ceiling.
        """
    from .perf import clamp_wait_ms, is_provisioning_wait

    # One ceiling, sized by one policy: `perf.max_wait_ms` is the maximum, and the caller's
    # measured think time may shorten it below that but never past it. The number is fed
    # *into* the existing clamp rather than enforced beside it — two clamps that disagree
    # is worse than one that is occasionally too tight, because the tighter one wins
    # silently and the looser one reads as a guarantee it is not.
    ceiling, mode = self._wait_ceiling()
    self._job_context.last_wait_ceiling_mode = mode
    if is_provisioning_wait(
        "job" if self._current_job_cancel_event() is not None else "observation"
    ):
        # `None` keeps meaning "no budget stated, use the ceiling" here too, so an exempt
        # caller and a clamped one disagree only about the number they were given.
        return (ceiling if requested_ms is None else int(requested_ms)), None, ceiling
    effective, was_clamped = clamp_wait_ms(requested_ms, self.config, ceiling_ms=ceiling)
    return effective, (int(requested_ms) if was_clamped and requested_ms else None), ceiling


def _sleep_between_polls(self: Engine, interval_ms: float, deadline: float) -> None:
    """Sleep until the next poll, but never past *deadline*.

        A bounded deadline buys nothing while one poll interval can outlast the whole budget:
        the loop checks the clock, sleeps out the interval, and only then notices it is late —
        so ``--interval 30000`` spends 30 seconds inside a 5-second ceiling. This is not a
        second ceiling, it is the existing one being enforced between two polls.
        """
    self._job_sleep(min(max(0.0, interval_ms) / 1000.0, max(0.0, deadline - time.monotonic())))


def _say_the_wait_was_shortened(
    self: Engine, result: ActionResult, clamped_from: int | None, ceiling: int
) -> ActionResult:
    """Record a clamp on the response, so 'not yet' cannot be read as 'not there'.

        Also names which policy produced the ceiling. Without it a caller cannot tell a number
        it could reproduce (`fixed`/`pinned`) from one that will move under it as its own
        latency is measured (`cold`/`adaptive`) — and a benchmark that cannot tell those apart
        is comparing two different budgets and calling it one.
        """
    if clamped_from is None:
        return result
    result.wait_clamped_from_ms = clamped_from
    result.wait_ceiling_ms = ceiling
    result.wait_ceiling_mode = getattr(self._job_context, "last_wait_ceiling_mode", None)
    hint = _wait_ceiling_explanation(clamped_from, ceiling)
    result.note = f"{result.note} {hint}".strip() if result.note else hint
    return result


def _wait_ceiling_explanation(clamped_from: int, ceiling: int) -> str:
    """The one sentence that explains a shortened wait, wherever it has to be said."""
    return (
        f"asked to wait {clamped_from}ms; capped at the {ceiling}ms ceiling "
        "(perf.max_wait_ms). The cap exists because an open-ended wait can be waiting for "
        "a change that already happened — call again and re-decide from a fresh read; a "
        "blocked session cannot."
    )


def _hint_for_a_shortened_wait(hint: str, clamped_from: int | None, ceiling: int) -> str:
    """The same explanation on a raising path, where there is no result to carry it.

        It goes *first* because it corrects the advice behind it: every one of these hints
        says "increase --timeout", which a clamped wait ignores.
        """
    if clamped_from is None:
        return hint
    return f"{_wait_ceiling_explanation(clamped_from, ceiling)} {hint}"


def _await_terms_on_observation(
    terms: list[_AwaitTerm],
    previous: list[dict[str, Any]],
    observation: AnalyzeResult,
    *,
    mode: MatchMode,
    ignore_case: bool,
) -> list[dict[str, Any]]:
    """Evaluate UI terms against one exact hierarchy frame.

        The ordinary poll uses ``Device.find_text`` because it is the cheapest possible check.
        Arrival-mismatch detection also needs to prove that the *stable frame* it inspected still
        misses the predicate.  Reusing results from an earlier selector RPC would combine two
        moments and could call a destination wrong while it was still rendering.

        Off-screen ``net:``/``log:`` terms retain their already evaluated value.  In practice the
        early mismatch path is intentionally disabled when a positive off-screen term is present,
        but retaining those rows keeps this helper total and the output order unchanged.
        """

    def matches(candidate: str, wanted: str) -> bool:
        hay = candidate.casefold() if ignore_case else candidate
        needle = wanted.casefold() if ignore_case else wanted
        if mode is MatchMode.exact:
            return hay == needle
        if mode is MatchMode.regex:
            flags = re.IGNORECASE if ignore_case else 0
            return re.search(wanted, candidate, flags) is not None
        return needle in hay

    refreshed: list[dict[str, Any]] = []
    for index, term in enumerate(terms):
        if term.by not in {"text", "rid", "desc"}:
            refreshed.append(dict(previous[index]))
            continue
        present = False
        for element in observation.elements:
            if term.by == "rid":
                full = element.resource_id or ""
                values = [full, _id_tail(full) or ""] if full else []
            elif term.by == "desc":
                values = [element.content_desc or ""]
            else:
                values = [element.text or "", element.content_desc or ""]
            if any(value and matches(value, term.value) for value in values):
                present = True
                break
        refreshed.append(
            {
                "term": term.text,
                "present": present,
                "satisfied": (not present) if term.negated else present,
            }
        )
    return refreshed


def _await_observation_identity(observation: AnalyzeResult) -> str | None:
    """Stable UI shape used only to confirm an action destination across fresh frames."""
    anchors = tuple(
        (
            _id_tail(element.resource_id) or "",
            element.content_desc or "",
            element.text or "",
            element.bounds,
        )
        for element in app_elements(observation.elements)
        if element.resource_id or element.content_desc or element.text
    )
    if not anchors:
        return None
    return hashlib.sha256(
        repr((observation.screen.package or "", anchors)).encode()
    ).hexdigest()[:16]


def _await_destination_changed(
    observation: AnalyzeResult, baseline: dict[str, Any] | None
) -> bool:
    """Whether a hierarchy frame is semantically different from the pre-action screen."""
    if baseline is None:
        return False
    before_identity = str(baseline.get("arrival_identity") or "")
    after_identity = _await_observation_identity(observation) or ""
    if before_identity and after_identity:
        return before_identity != after_identity
    before_package = str(baseline.get("package") or "")
    after_package = str(observation.screen.package or "")
    if before_package and after_package and before_package != after_package:
        return True
    before_known = str(baseline.get("known_screen") or "")
    after_known = str(observation.meta.known_screen or "")
    if before_known and after_known and before_known != after_known:
        return True
    before_labels = {str(value) for value in baseline.get("labels") or [] if value}
    after_labels = {
        _label(value)
        for element in app_elements(observation.elements)
        for value in (element.text, element.content_desc)
        if value and _label(value)
    }
    if before_labels != after_labels:
        return True
    before_rids = {str(value) for value in baseline.get("rids") or [] if value}
    after_rids = {
        rid
        for element in app_elements(observation.elements)
        if (rid := _id_tail(element.resource_id))
    }
    if before_rids and before_rids != after_rids:
        return True
    return int(baseline.get("count") or 0) != len(observation.elements)


def _arrival_predicate_suggestions(
    observation: AnalyzeResult,
    baseline: dict[str, Any] | None,
    *,
    limit: int = 3,
) -> list[str]:
    """Stable positive predicates visible only after (or at least on) the destination.

        Resource ids are preferred because they survive copy changes and do not echo user content.
        Text/description is a fallback for apps that expose no ids.  Numeric frame ids are never
        suggested: they are observation-local and are exactly what this recovery is meant to make
        unnecessary.
        """

    def escaped(value: str) -> str:
        return value.replace("\\", "\\\\").replace(",", "\\,")

    before_rids = {str(value) for value in (baseline or {}).get("rids") or [] if value}
    before_labels = {str(value) for value in (baseline or {}).get("labels") or [] if value}
    elements = app_elements(observation.elements)
    # Actionable controls first, then the remaining anchors in visual order.
    ordered = sorted(
        elements,
        key=lambda element: (
            0 if element.enabled and (element.clickable or element.checkable) else 1,
            element.bounds[1],
            element.bounds[0],
        ),
    )
    suggestions: list[str] = []
    seen: set[str] = set()

    def add(prefix: str, value: str) -> None:
        value = _label(value)
        if not value or len(value) > 120:
            return
        predicate = f"{prefix}:{escaped(value)}"
        key = predicate.casefold()
        if key not in seen:
            seen.add(key)
            suggestions.append(predicate)

    # Prefer anchors introduced by the action, then fall back to any destination anchor.
    for new_only in (True, False):
        for element in ordered:
            rid = _id_tail(element.resource_id)
            if not rid or (new_only and rid in before_rids):
                continue
            add("rid", rid)
            if len(suggestions) >= limit:
                return suggestions
    for new_only in (True, False):
        for element in ordered:
            if element.password:
                continue
            for prefix, value in (
                ("text", element.text or ""),
                ("desc", element.content_desc or ""),
            ):
                label = _label(value)
                if not label or label.isdigit() or (new_only and label in before_labels):
                    continue
                add(prefix, label)
                if len(suggestions) >= limit:
                    return suggestions
    return suggestions


def _sample_action_destination(self: Engine) -> AnalyzeResult | None:
    """One fresh, hierarchy-only frame for action-arrival mismatch detection."""
    try:
        return self.analyze(
            source="hierarchy",
            with_ocr=False,
            no_cache=True,
            record=False,
        )
    except Exception as exc:  # noqa: BLE001 - a missed optimization must not fail the wait
        logger.debug("action arrival sample unavailable: %s", exc)
        return None


def await_predicate(
    self: Engine,
    predicate: str,
    *,
    timeout_ms: int = 60_000,
    poll_ms: int = 500,
    match: str = "contains",
    ignore_case: bool = False,
    observe: bool = False,
    adopt_action: bool = False,
    rich_ui: bool = True,
    hierarchy_only: bool = False,
) -> ActionResult:
    """Wait until *predicate* holds, and say exactly what ended the wait.

        A long-running synthetic export demonstrates the ambiguity: without a condition to wait
        *on*, a caller can only poll, wait fixed intervals, or conclude "stuck" from a stale frame.
        The output must distinguish a hang from a slow backend, so the outcome is a named field
        rather than something inferred from `ok`:

        * ``satisfied`` — every term held.
        * ``screen-changed`` — the foreground activity or package moved while waiting and the
          predicate is still unmet. Returns immediately instead of burning the budget: the
          surface being waited on is gone, so more waiting cannot help. This is the outcome that
          separates "we got kicked out / an error dialog took over" from "still working".
        * ``settled-unmet`` — action-bound waits only: the action reached a stable, non-loading,
          semantically different destination in the same activity, but the caller's positive UI
          arrival term is not on it. This returns a structured ``arrival_mismatch`` rather than
          spending a long budget on a predicate that describes the screen left behind.
        * ``timeout`` — budget spent, predicate unmet, still on the same screen.

        **Not** network idle. A sample app may prefetch, post telemetry, or stream status updates
        continuously, so idleness is a flaky proxy for "this is ready". A predicate says what is
        actually wanted.

        Standalone ``screen-changed`` remains keyed on the resumed activity/package and
        deliberately not on the element tree: a streaming surface rewrites its tree constantly,
        so a tree-change trigger would abort every legitimate wait on exactly the screens this
        exists for. The stable-tree check is reachable only when ``adopt_action`` says one action
        has already run, and requires two equal fresh destination frames.

        Per-term results are always returned, satisfied or not, because *which* term is missing is
        how a reader tells a failed load from a slow one: spinner gone but results absent is a
        failure, spinner still present is progress.
        """
    # One ceiling for every observation wait, this one included. It was missed here at
    # first and a single `await-and-analyze` then ran 62s in a live pass — the default was
    # 60s and nothing capped it. `await` is the wait an agent should reach for most, so an
    # uncapped default here undoes the ceiling everywhere else.
    timeout_ms, _await_clamped_from, _await_ceiling = self._bounded_wait_ms(timeout_ms)
    # `await` is the one wait whose clamp was computed and then dropped: the budget was
    # shortened correctly but the response never said so, which is exactly the reading error
    # the ceiling machinery exists to prevent — "predicate unmet" after a silently trimmed
    # wait is indistinguishable from "predicate will never hold". Handed to `_await_result`
    # through the engine because the outcome is built four call sites deep.
    self._pending_wait_clamp = (_await_clamped_from, _await_ceiling)

    terms = _parse_await_terms(predicate, require_positive=adopt_action)
    device = self.device
    mode = MatchMode(match)
    action_baseline = deepcopy(self._action_observation_baseline) if adopt_action else None
    positive_terms = [term for term in terms if not term.negated]
    # Which name a fully-held predicate earns. An absence-only predicate holding proves the
    # screen the caller left is gone; it says nothing about where it landed, and the two must
    # not be reported under one name. `adopt_action` cannot reach this branch — it still
    # requires a positive term above — so this only ever renames a *standalone* await, which
    # main had been calling `satisfied` on strictly weaker evidence.
    held_outcome = "satisfied" if positive_terms else "absence-satisfied"
    detect_arrival_mismatch = bool(
        adopt_action
        and action_baseline is not None
        and positive_terms
        # A stable UI cannot prove that an asynchronous network/log event will never arrive.
        # Preserve those waits rather than turning a quiet screen into a false mismatch.
        and all(term.by in {"text", "rid", "desc"} for term in positive_terms)
    )
    stable_destination_identity: str | None = None
    stable_destination_checks = 0

    def snapshot() -> tuple[str, str]:
        try:
            info = device.current_app() or {}
        except Exception:  # a device hiccup must not be read as a navigation
            return ("", "")
        return (str(info.get("package") or ""), str(info.get("activity") or ""))

    # Baseline for the off-screen terms, taken before the first evaluation so a `net:` /
    # `log:` term only ever matches evidence produced *after* the wait began. Without it
    # the previous turn's response would satisfy this turn's wait instantly.
    wall_baseline = time.time()

    def _log_baseline_ms() -> int:
        """Baseline as a **device**-clock epoch — `logcat(since_ms=…)` demands one.

            The host clock is not interchangeable: an emulator can sit seconds off, and a
            baseline in the wrong frame either drops the very lines we are waiting for or
            admits the previous turn's. The measured skew is cached, so this costs no adb
            round-trip on the poll path.
            """
        try:
            from . import logcat as logcat_mod

            clock = logcat_mod.resolve_clock(device, self.config.cache.dir)
            return int(clock.to_device(int(wall_baseline * 1000)))
        except Exception:
            return int(wall_baseline * 1000)

    log_baseline_ms = _log_baseline_ms() if any(t.by == "log" for t in terms) else 0

    def _net_present(spec: str) -> bool:
        try:
            proxy_mock = self.platform.capability("proxy")

            flows = proxy_mock.read_flows_since(
                self.config.cache.dir, wall_baseline, self._proxy_serial()
            )
        except Exception:  # proxy not running / extra not installed
            return False
        return any(proxy_mock.flow_matches(f, spec) for f in flows)

    def _log_present(spec: str) -> bool:
        try:
            lines = device.logcat(dump=True, since_ms=log_baseline_ms) or ""
        except TypeError:  # device implementations that take no since filter
            try:
                lines = device.logcat(dump=True) or ""
            except Exception:
                return False
        except Exception:
            return False
        if not isinstance(lines, str):
            lines = "\n".join(str(x) for x in lines)
        haystack = lines.lower() if ignore_case else lines
        needle = spec.lower() if ignore_case else spec
        return needle in haystack

    def evaluate() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for term in terms:
            if term.by == "net":
                present = _net_present(term.value)
            elif term.by == "log":
                present = _log_present(term.value)
            else:
                present = (
                    device.find_text(
                        term.value, match=mode, ignore_case=ignore_case, by=term.by
                    )
                    is not None
                )
            out.append(
                {
                    "term": term.text,
                    "present": present,
                    "satisfied": (not present) if term.negated else present,
                }
            )
        return out

    ui_terms = [term for term in terms if term.by in {"text", "desc"}]

    def evaluate_rich() -> list[dict[str, Any]] | None:
        """Verify UI text against hierarchy plus OCR before making a final claim.

            The device selector only sees accessibility text. That is cheap enough to poll, but
            it made ``!text:Loading`` succeed immediately on a visible canvas label and made a
            positive result time out even though OCR could read it. Rich verification is bounded:
            once before accepting a negated UI term and once at the deadline for a positive miss.
            """
        if not rich_ui or not ui_terms:
            return None
        try:
            observed = self.analyze(source="hierarchy", with_ocr=True, record=False)
        except Exception:  # noqa: BLE001 - unavailable OCR preserves hierarchy semantics
            return None
        base_present = {str(result["term"]): bool(result["present"]) for result in results}

        def matches(value: str, needle: str) -> bool:
            candidate = value.casefold() if ignore_case else value
            wanted = needle.casefold() if ignore_case else needle
            if mode is MatchMode.exact:
                return candidate == wanted
            if mode is MatchMode.regex:
                flags = re.IGNORECASE if ignore_case else 0
                return re.search(needle, value, flags) is not None
            return wanted in candidate

        rich: list[dict[str, Any]] = []
        for term in terms:
            if term.by not in {"text", "desc"}:
                present = base_present.get(term.text, False)
            else:
                values: list[str] = []
                for element in observed.elements:
                    if term.by == "text":
                        values.extend(
                            value for value in (element.text, element.content_desc) if value
                        )
                    elif element.content_desc:
                        values.append(element.content_desc)
                # Rich analysis is an enrichment, never a replacement: a provider may return
                # only OCR boxes while the cheap selector already proved a hierarchy term.
                present = base_present.get(term.text, False) or any(
                    matches(value, term.value) for value in values
                )
            rich.append(
                {
                    "term": term.text,
                    "present": present,
                    "satisfied": (not present) if term.negated else present,
                }
            )
        return rich

    started_at = time.monotonic()
    # Internal hierarchy-only navigation already obtains a fresh observation before the
    # next action. Android's `app_current` is unexpectedly expensive on some devices
    # (~5s per call), and polling it before/after each 1.2s Back step made the "bounded"
    # primitive take 31s. Package boundaries are verified from that observation by
    # `back_until`, so omit redundant activity RPCs on this private fast path.
    origin = ("", "") if hierarchy_only else snapshot()
    deadline = started_at + max(0.0, timeout_ms / 1000.0)
    next_negative_rich_at = started_at
    negative_ui_terms = any(term.negated for term in ui_terms)
    checks = 0
    self._job_checkpoint()
    results = evaluate()
    while True:
        self._job_checkpoint()
        checks += 1
        if all(t["satisfied"] for t in results):
            if not negative_ui_terms:
                return self._await_result(
                    held_outcome,
                    results,
                    started_at,
                    checks,
                    origin,
                    origin,
                    observe,
                    adopt_action,
                    hierarchy_only=hierarchy_only,
                    capture_terms=terms,
                )
            # A negated accessibility miss is not proof of visual absence. Verify with OCR,
            # but at most every two seconds while a canvas/loading label remains visible.
            if time.monotonic() >= next_negative_rich_at:
                rich = evaluate_rich()
                next_negative_rich_at = time.monotonic() + max(2.0, poll_ms / 250.0)
                if rich is None or all(term["satisfied"] for term in rich):
                    return self._await_result(
                        held_outcome,
                        rich or results,
                        started_at,
                        checks,
                        origin,
                        origin,
                        observe,
                        adopt_action,
                        hierarchy_only=hierarchy_only,
                        capture_terms=terms,
                    )
                results = rich
        if detect_arrival_mismatch:
            destination = self._sample_action_destination()
            if destination is not None:
                destination_terms = self._await_terms_on_observation(
                    terms,
                    results,
                    destination,
                    mode=mode,
                    ignore_case=ignore_case,
                )
                if all(term["satisfied"] for term in destination_terms):
                    return self._await_result(
                        "satisfied",
                        destination_terms,
                        started_at,
                        checks,
                        origin,
                        origin,
                        observe,
                        adopt_action,
                        hierarchy_only=hierarchy_only,
                        capture_terms=terms,
                    )
                unmet_positive = [
                    row["term"]
                    for term, row in zip(terms, destination_terms, strict=True)
                    if not term.negated and not row["satisfied"]
                ]
                negative_unmet = any(
                    term.negated and not row["satisfied"]
                    for term, row in zip(terms, destination_terms, strict=True)
                )
                identity = self._await_observation_identity(destination)
                candidate = bool(
                    unmet_positive
                    and not negative_unmet
                    and identity
                    and not self._observation_is_loading(destination)
                    and self._await_destination_changed(destination, action_baseline)
                )
                if candidate:
                    if identity == stable_destination_identity:
                        stable_destination_checks += 1
                    else:
                        stable_destination_identity = identity
                        stable_destination_checks = 1
                    if stable_destination_checks >= 2:
                        suggestions = self._arrival_predicate_suggestions(
                            destination,
                            action_baseline,
                        )
                        satisfied_negatives = [
                            term.text
                            for term, row in zip(terms, destination_terms, strict=True)
                            if term.negated and row["satisfied"]
                        ]
                        corrected = ",".join([*suggestions[:1], *satisfied_negatives])
                        recommended_call = (
                            f"aua await-and-analyze {shlex.quote(corrected)} --observe"
                            if corrected
                            else None
                        )
                        mismatch: dict[str, Any] = {
                            "code": "arrival_mismatch",
                            "original_predicate": predicate,
                            "unmet_positive_terms": unmet_positive,
                            "suggested_positive_predicates": suggestions,
                            "stable_checks": stable_destination_checks,
                            "screen_changed": True,
                            "loading": False,
                            "action_repeated": False,
                        }
                        if destination.meta.known_screen:
                            mismatch["known_screen"] = destination.meta.known_screen
                        if recommended_call:
                            mismatch["recommended_call"] = recommended_call
                            mismatch["recommended_mcp_call"] = {
                                "tool": "await_and_analyze",
                                "arguments": {"predicate": corrected},
                            }
                        return self._await_result(
                            "settled-unmet",
                            destination_terms,
                            started_at,
                            checks,
                            origin,
                            origin,
                            observe,
                            adopt_action,
                            hierarchy_only=hierarchy_only,
                            arrival_mismatch=mismatch,
                            capture_terms=terms,
                        )
                else:
                    stable_destination_identity = None
                    stable_destination_checks = 0
        now = origin if hierarchy_only else snapshot()
        if now != origin and any(now):
            return self._await_result(
                "screen-changed",
                results,
                started_at,
                checks,
                origin,
                now,
                observe,
                adopt_action,
                hierarchy_only=hierarchy_only,
                capture_terms=terms,
            )
        if time.monotonic() >= deadline:
            rich = evaluate_rich()
            if rich is not None and all(term["satisfied"] for term in rich):
                return self._await_result(
                    held_outcome,
                    rich,
                    started_at,
                    checks,
                    origin,
                    origin,
                    observe,
                    adopt_action,
                    hierarchy_only=hierarchy_only,
                    capture_terms=terms,
                )
            return self._await_result(
                "timeout",
                results,
                started_at,
                checks,
                origin,
                now,
                observe,
                adopt_action,
                hierarchy_only=hierarchy_only,
                capture_terms=terms,
            )
        self._sleep_between_polls(max(10.0, float(poll_ms)), deadline)
        results = evaluate()


def _unknown_map_selectors(self: Engine, unmet: list[str], package: str) -> list[dict[str, Any]]:
    """Unmet ``rid:`` terms this app's map has never recorded on any screen.

        Deliberately map-based rather than screen-based: "not on this screen" is what an unmet
        term already says. What an agent cannot see, and what sends it inventing a second id, is
        that the id exists nowhere in the app. Not cached — it runs only when a positive term
        went unmet, and a map that just learned a screen must not be answered from a stale copy.
        """
    if not unmet or not package or self._memory is None:
        return []
    app = self._memory.load(package)
    if app is None:
        return []
    from .selectors import unknown_map_rids

    vocabulary = {
        anchor[3:]
        for screen in app.screens.values()
        for anchor in screen.anchors
        if anchor.startswith("id:")
    }
    return unknown_map_rids(unmet, vocabulary)


def _await_result(
    self: Engine,
    outcome: str,
    terms: list[dict[str, Any]],
    started_at: float,
    checks: int,
    origin: tuple[str, str],
    now: tuple[str, str],
    observe: bool,
    adopt_action: bool = False,
    *,
    hierarchy_only: bool = False,
    arrival_mismatch: dict[str, Any] | None = None,
    capture_terms: list[_AwaitTerm] | None = None,
) -> ActionResult:
    elapsed = int((time.monotonic() - started_at) * 1000)
    unmet = [t["term"] for t in terms if not t["satisfied"]]
    detail = f"{outcome} after {elapsed}ms ({checks} checks)"
    if unmet:
        detail += "; unmet: " + ", ".join(unmet)
    # An unmet id that no mapped screen of this app has ever carried is a caller mistake,
    # not a slow load, and the two are indistinguishable from "unmet:" alone. Checked only
    # on the unmet path, so a satisfied wait pays nothing.
    impossible = self._unknown_map_selectors(unmet, now[0] or origin[0]) if unmet else []
    if impossible:
        detail += "; " + ", ".join(
            f"{row['term']} is in no mapped screen of this app"
            + (f" (nearest: {', '.join(row['nearest'])})" if row["nearest"] else "")
            for row in impossible
        )
    if outcome == "screen-changed":
        detail += f"; now on {now[0]}/{now[1]} (was {origin[0]}/{origin[1]})"
    result = ActionResult(
        ok=outcome in _AWAIT_PREDICATE_HELD,
        action="await",
        detail=detail,
        # `outcome` rides in `acting`-style structured form so a caller branches on a field
        # rather than parsing prose. `ok` alone cannot carry three states.
        await_outcome=outcome,
        await_terms=terms,
        arrival_mismatch=arrival_mismatch,
        unknown_selectors=impossible or None,
        elapsed_ms=elapsed,
    )
    if outcome == "absence-satisfied":
        # `ok` is true because the wait did exactly what it was asked to. The caveat is the
        # part `ok` cannot carry: nothing here evidences the destination, so a caller that
        # needs one must still name it.
        result.note = (
            "every term held, but they were all absence terms: what you left is gone and "
            "nothing here proves what arrived. Read `observation` to see where you landed, "
            "then wait on a positive `text:`/`rid:`/`desc:` term from it if arrival matters."
        )
    # A standalone await is read-only. A global action ``--until`` is different: its final
    # evidence replaces the action's early loading-shell readback, so it must run the normal
    # recording path and consume the still-pending action into this destination. The CLI
    # opts into this explicitly; MCP/standalone waits retain their passive behaviour.
    observed = self._observe(
        result,
        observe,
        settle=False,
        # A timeout is explicitly not final evidence; recording it would merely replace an
        # early loading shell with a later loading shell and consume the action anyway.
        record_screen=adopt_action and outcome != "timeout",
        hierarchy_only=hierarchy_only,
        adopt_action=adopt_action,
    )
    if (
        adopt_action
        and outcome == "satisfied"
        and observed.observation is not None
        and capture_terms
    ):
        memory = self._memory
        if memory is not None and self._join_memory_writers(timeout_s=5.0):
            with contextlib.suppress(Exception):
                memory.record_action_arrival(
                    self.device.serial,
                    terms=[
                        {
                            "by": term.by,
                            "value": term.value,
                            "negated": term.negated,
                        }
                        for term in capture_terms
                    ],
                    fingerprint=observed.observation.meta.fingerprint,
                    package=observed.observation.screen.package,
                )
    if arrival_mismatch is not None:
        call = arrival_mismatch.get("recommended_call")
        observed.note = (
            "The action ran once and reached this stable destination, but its arrival "
            "predicate names content that is not here. Reuse this fresh observation and do "
            "not repeat the action."
        )
        if call:
            observed.note += f" If explicit validation is needed, use `{call}`."
    clamp = self._pending_wait_clamp
    # Consume-once: the engine outlives one command under the warm daemon, and a leftover
    # clamp would tell a later, unclamped wait that its budget had been cut.
    self._pending_wait_clamp = None
    if clamp is not None:
        observed = self._say_the_wait_was_shortened(observed, clamp[0], clamp[1])
    return observed


def wait(
    self: Engine,
    *,
    for_: str | None = None,
    idle: bool = False,
    timeout_ms: int = 5000,
    match: str = "contains",
    ignore_case: bool = False,
    observe: bool = False,
    by: str = "text",
    absent: bool = False,
) -> ActionResult:
    """Wait for text to appear or disappear, or for the UI to go idle.

        ``timeout_ms`` is sized by :meth:`_bounded_wait_ms` before anything blocks on it. This
        was the last agent-facing wait handing the caller's budget straight to the device, so
        `wait-and-analyze --for X --timeout-ms 120000` blocked for two minutes and made the
        ceiling on its sibling waits meaningless.
        """
    self._start_call()
    device = self.device
    timeout_ms, clamped_from, ceiling_ms = self._bounded_wait_ms(timeout_ms)
    if idle:
        device.wait_idle(timeout_ms)
        return self._say_the_wait_was_shortened(
            self._observe(
                ActionResult(ok=True, action="wait", detail="idle"), observe, settle=False
            ),
            clamped_from,
            ceiling_ms,
        )
    if not for_:
        raise UsageError("wait needs --for <text> or --idle")
    for_, by, absent = _parse_wait_for_predicate(for_, by=by, absent=absent)
    mode = MatchMode(match)
    candidates = self._locale_candidates(device, for_, by)
    if absent:
        # Wait until the target is NO LONGER present (loading spinners, transient
        # dialogs) — Maestro's `notVisible`. ok=True once it's gone. Known translated
        # renderings count as present too, else a source-language spinner label
        # reports gone while its device-locale rendering is still on screen.
        probes = [for_] + [c for c, _, _ in candidates]
        deadline = time.monotonic() + timeout_ms / 1000.0
        gone = False
        while True:
            if all(
                device.find_text(p, match=mode, ignore_case=ignore_case, by=by) is None
                for p in probes
            ):
                gone = True
                break
            if time.monotonic() >= deadline:
                break
            self._sleep_between_polls(200.0, deadline)
        if not gone:
            detail = self._wait_timeout_message(
                for_, mode=mode, by=by, ignore_case=ignore_case, absent=True
            )
            return self._say_the_wait_was_shortened(
                self._observe(
                    ActionResult(ok=False, action="wait", detail=detail), observe, settle=False
                ),
                clamped_from,
                ceiling_ms,
            )
        return self._say_the_wait_was_shortened(
            self._observe(
                ActionResult(ok=True, action="wait", detail=f"absent:{for_}"),
                observe,
                settle=False,
            ),
            clamped_from,
            ceiling_ms,
        )
    found, matched = self._wait_for_any(
        device,
        for_,
        candidates,
        mode=mode,
        ignore_case=ignore_case,
        timeout_ms=timeout_ms,
        by=by,
    )
    if found is None:
        detail = self._wait_timeout_message(
            for_, mode=mode, by=by, ignore_case=ignore_case, absent=False
        )
        return self._say_the_wait_was_shortened(
            self._observe(
                ActionResult(ok=False, action="wait", detail=detail), observe, settle=False
            ),
            clamped_from,
            ceiling_ms,
        )
    result = ActionResult(
        ok=True,
        action="wait",
        detail=for_,
        target=list(found),
        hint=self._translated_hint(matched[0], matched[1], matched[2], for_)
        if matched
        else None,
    )
    # `--observe` returns the screen with fresh ids so the agent acts without a separate
    # `analyze` — attached even on a MISS, so a failed wait is diagnosable in one call.
    # settle=False: wait already blocked on the condition; don't pay pixel-settle again.
    return self._say_the_wait_was_shortened(
        self._observe(result, observe, settle=False), clamped_from, ceiling_ms
    )


def _journal_wait_gave_up(self: Engine, kind: str, detail: str) -> None:
    """Log a wait that ended by raising, before the exception leaves.

        The successful path is journalled on the way out of `_observe`; a timeout never gets
        there. Leaving it unrecorded would hide exactly the wrong number — the one wait that
        spent its entire budget.
        """
    self._journal_call_answer(
        ActionResult(ok=False, action=kind, detail=detail, elapsed_ms=self._wall_ms()),
        outcome="timeout",
    )


def hierarchy_fingerprint(self: Engine, *, background: bool = False) -> str | None:
    """Cheap SHA1 of the current hierarchy dump (no parse). Used by watch/push.

        The push watcher is a long-lived background thread. It may observe through the shared
        activity fence, but it must neither connect nor retain the foreground command mutex.
        """
    device = self._device if background else self.device
    if device is None:
        return None
    compressed = bool(self.config.device.compressed_hierarchy)
    try:
        if background:
            with self.device_use_context(device.serial):
                xml = self.platform.dump_tree(device, compact=compressed)
        else:
            xml = self.platform.dump_tree(device, compact=compressed)
    except Exception:  # pragma: no cover
        return None
    return hashlib.sha1(xml.encode()).hexdigest()


def wait_changed(
    self: Engine,
    *,
    timeout_ms: int = 15000,
    interval_ms: int | None = None,
    observe: bool = False,
) -> ActionResult:
    """Block until the hierarchy fingerprint changes (or timeout).

        Host-polled stand-in for AccessibilityEvent push (phase 2). Prefer this over
        busy ``analyze`` loops when waiting for *any* UI change.

        ``timeout_ms`` is sized by :meth:`_bounded_wait_ms`: "any change" is the weakest thing
        to wait on, so it is the wait most worth cutting short — a caller with 60s to spend
        should be waiting on evidence with ``--until`` instead.
        """
    self._start_call()
    interval = (
        interval_ms if interval_ms is not None else int(self.config.daemon.watch_interval_ms)
    )
    baseline = self.hierarchy_fingerprint()
    timeout_ms, clamped_from, ceiling_ms = self._bounded_wait_ms(timeout_ms)
    deadline = time.monotonic() + timeout_ms / 1000.0
    samples = 0
    while time.monotonic() < deadline:
        self._sleep_between_polls(max(50.0, float(interval)), deadline)
        samples += 1
        self._job_checkpoint()
        fp = self.hierarchy_fingerprint()
        if fp and baseline and fp != baseline:
            return self._say_the_wait_was_shortened(
                self._observe(
                    ActionResult(
                        ok=True,
                        action="wait-changed",
                        detail=f"changed after {samples} samples fingerprint={fp[:12]}",
                    ),
                    observe,
                    settle=False,
                ),
                clamped_from,
                ceiling_ms,
            )
        if fp and baseline is None:
            baseline = fp
    self._journal_wait_gave_up(
        "wait-changed",
        f"hierarchy did not change within {timeout_ms} ms ({samples} samples)",
    )
    raise StabilityTimeout(
        f"hierarchy did not change within {timeout_ms} ms ({samples} samples)",
        hint=self._hint_for_a_shortened_wait(
            "Increase --timeout, or the screen is idle. "
            "Use `aua wait-and-analyze --for` for a label.",
            clamped_from,
            ceiling_ms,
        ),
    )


def wait_after_change(
    self: Engine,
    *,
    timeout_ms: int = 60_000,
    interval_ms: int = 120,
    settle_ms: int = 1_200,
    confirmation_ms: int = 1_800,
    observe: bool = False,
) -> ActionResult:
    """Wait for a change, visual settle, and a bounded late-change confirmation.

        A loading shell can become visually quiet while its request is still running. Plainly
        composing :meth:`wait_changed` with :meth:`wait_stable` therefore accepts the first quiet
        spinner frame as the result. This contract adds a second, bounded phase: after visual
        settle, the hierarchy must stay unchanged for ``confirmation_ms``. If later content lands
        during that window, stability is measured again from the new frame.

        The confirmation uses the hierarchy rather than full-frame pixels so a looping spinner or
        video remains maskable by :meth:`wait_stable`. Opaque/canvas results should still use an
        explicit predicate, which is the only generic proof that particular content arrived.
        ``timeout_ms`` bounds the complete change + settle + confirmation sequence.
        """
    timeout_ms, clamped_from, ceiling = self._bounded_wait_ms(timeout_ms)
    started = self._start_call()
    deadline = started + max(0.0, timeout_ms / 1000.0)

    def remaining_ms() -> int:
        return max(1, int((deadline - time.monotonic()) * 1000))

    # The change may already have landed while the caller was composing this call — which
    # in the field is exactly what happened, and waiting for the *next* one cost 41s on a
    # screen that had been ready the whole time. A settled screen with content on it is the
    # answer, so take it rather than blocking for a repeat.
    if self._screen_already_answers():
        return self._say_the_wait_was_shortened(
            self._observe(
                ActionResult(
                    ok=True,
                    action="wait-after-change",
                    detail=(
                        "already settled with content on screen — returned without waiting "
                        "for a further change"
                    ),
                ),
                observe,
                settle=False,
            ),
            clamped_from,
            ceiling,
        )
    with contextlib.suppress(StabilityTimeout):
        self.wait_changed(
            timeout_ms=remaining_ms(),
            interval_ms=interval_ms,
            observe=False,
        )
    late_changes = 0
    while True:
        if time.monotonic() >= deadline:
            return self._hand_back_what_is_on_screen(
                action="wait-after-change",
                waited_ms=int((time.monotonic() - started) * 1000),
                ceiling_ms=timeout_ms,
                observe=observe,
                clamped_from=clamped_from,
            )
        try:
            self.wait_stable(
                interval_ms=interval_ms,
                settle_ms=max(1, settle_ms),
                timeout_ms=remaining_ms(),
                observe=False,
            )
        except StabilityTimeout:
            # The inner settle running out is the same event as the outer deadline: the
            # screen is still moving. It is the caller's budget that expired, not a device
            # fault, so hand back what is on screen instead of raising through a bounded
            # wait the caller was told to expect to expire.
            return self._hand_back_what_is_on_screen(
                action="wait-after-change",
                waited_ms=int((time.monotonic() - started) * 1000),
                ceiling_ms=timeout_ms,
                observe=observe,
                clamped_from=clamped_from,
            )

        baseline = self.hierarchy_fingerprint()
        confirm_deadline = min(
            deadline,
            time.monotonic() + max(0, confirmation_ms) / 1000.0,
        )
        changed_again = False
        while time.monotonic() < confirm_deadline:
            self._sleep_between_polls(max(10.0, float(interval_ms)), confirm_deadline)
            current = self.hierarchy_fingerprint()
            if baseline and current and current != baseline:
                changed_again = True
                late_changes += 1
                break
            if baseline is None and current:
                baseline = current
        if changed_again:
            continue
        if confirm_deadline >= deadline and confirmation_ms > 0:
            return self._hand_back_what_is_on_screen(
                action="wait-after-change",
                waited_ms=int((time.monotonic() - started) * 1000),
                ceiling_ms=timeout_ms,
                observe=observe,
                clamped_from=clamped_from,
            )
        elapsed = int((time.monotonic() - started) * 1000)
        detail = f"changed and confirmed settled after {elapsed}ms"
        if late_changes:
            detail += f" ({late_changes} late change(s) restabilized)"
        return self._say_the_wait_was_shortened(
            self._observe(
                ActionResult(ok=True, action="wait-after-change", detail=detail),
                observe,
                settle=False,
            ),
            clamped_from,
            ceiling,
        )


def _wait_timeout_message(
    self: Engine,
    needle: str,
    *,
    mode: MatchMode,
    by: str,
    ignore_case: bool,
    absent: bool,
) -> str:
    """Rich timeout diagnosis — mode, fields, candidates, accidental-regex hint."""
    field = {"text": "text", "id": "resource-id", "desc": "content-desc"}.get(by, by)
    intent = "still present" if absent else "never appeared"
    parts = [
        f"wait timed out: {needle!r} {intent} "
        f"(match={mode.value}, by={by}, fields={field}"
        f"{', ignore_case' if ignore_case else ''})"
    ]
    # Accidental regex under contains — an observed agent failure mode.
    meta = set(r".*+?[](){}|^$\\")
    if mode is MatchMode.contains and any(c in needle for c in meta):
        parts.append(
            f"hint: pattern looks like regex but --match is '{mode.value}' "
            f"(matched literally as a substring). Use --match regex."
        )
    # A label written in another language than the device renders never matches.
    if not absent:
        locale_part = self._text_miss_hint(self.device, by, needle, tried_translations=True)
        if locale_part:
            parts.append(locale_part)
    # Closest on-screen candidates.
    try:
        result = self.analyze(source="hierarchy", record=False)
        from .selectors import app_elements, nearest_elements

        near = nearest_elements(result.elements, needle, limit=5)
        if near:
            digests = []
            for el in near:
                label = el.text or el.content_desc or (el.resource_id or "").split("/")[-1]
                digests.append(f"id={el.id}:{label!r}")
            parts.append("closest on screen: " + "; ".join(digests))
        else:
            app_count = len(app_elements(result.elements))
            parts.append(f"screen has {app_count} app elements (no close text match)")
    except Exception as exc:  # pragma: no cover - diagnostic bonus
        parts.append(f"(could not snapshot screen: {exc})")
    return " — ".join(parts)


def _node_state(self: Engine, xml: str, el: Element) -> dict[str, Any]:
    """Interaction state from the selected platform's native-tree adapter."""

    return dict(self.platform.element_state(xml, el))


def _check_predicates(
    self: Engine, el: Element, state: dict[str, Any], predicates: dict[str, Any]
) -> list[str]:
    """Names of the predicates that do NOT hold, as ``expected!=actual`` strings."""
    labels = [v for v in (state["text"], state["content_desc"]) if v]
    failures: list[str] = []
    for name, want in predicates.items():
        if name in ("exists", "absent"):
            continue
        if name == "text_is":
            if not any(v.strip() == want for v in labels):
                failures.append(f"text_is={want!r}!=actual={labels or None!r}")
        elif name == "text_contains":
            if not any(want.lower() in v.lower() for v in labels):
                failures.append(f"text_contains={want!r}!=actual={labels or None!r}")
        else:
            actual = state.get(name)
            if bool(actual) is not bool(want):
                failures.append(f"{name}={str(want).lower()}!=actual={str(actual).lower()}")
    return failures


def _expect_once(
    self: Engine,
    selector: dict[str, Any],
    predicates: dict[str, Any],
    *,
    index: int | None,
    first: bool,
    count: int | None = None,
    within: Selector | None = None,
    same_parent_as: Selector | None = None,
    contains_all: Sequence[Selector] = (),
) -> tuple[bool, str]:
    """One evaluation pass: ``(ok, detail)``. One hierarchy dump, no screenshots."""
    xml = self.platform.dump_tree(self.device)
    w, h = self.device.window_size()
    elements = self.platform.normalize_tree(
        xml,
        (w, h),
        ignored_app_ids=self.config.memory.ignore_packages,
    ).elements
    label = selector_label(selector)
    matches = match_selector(elements, **selector)
    structural = apply_structural_filters(
        elements,
        matches,
        within=within,
        same_parent_as=same_parent_as,
    )
    if not structural.ok:
        return False, detail_tokens("fail", sought=label) + " | " + str(structural.detail)
    matches = list(structural.matches)
    if count is not None and len(matches) != count:
        detail = detail_tokens(
            "fail",
            sought=label,
            predicate="count",
            expected=count,
            actual=len(matches),
        )
        if matches:
            detail += " | found: " + " | ".join(
                element_digest(el) for el in matches[:_MAX_CANDIDATES]
            )
        return False, detail
    if count == 0:
        return True, detail_tokens(
            "pass", sought=label, predicate="count", expected=0, actual=0
        )
    if predicates.get("absent"):
        if not matches:
            return True, detail_tokens("pass", sought=label, predicate="absent")
        return False, detail_tokens(
            "fail", sought=label, predicate="absent", actual="present"
        ) + " | found: " + " | ".join(element_digest(el) for el in matches[:_MAX_CANDIDATES])
    if not matches:
        near = nearest_elements(elements, selector.get("rid") or selector.get("text") or "")
        app_only = app_elements(elements)
        detail = detail_tokens(
            "fail",
            sought=label,
            predicate="exists",
            actual="absent",
            on_screen=len(app_only),
            system=len(elements) - len(app_only) or None,
        )
        if near:
            detail += " | nearest: " + " | ".join(element_digest(el) for el in near)
        return False, detail
    state_only = [k for k in predicates if k not in ("exists", "absent")]
    structural_only = bool(within or same_parent_as or contains_all)
    if len(matches) > 1 and (state_only or structural_only) and index is None and not first:
        raise SelectorAmbiguousError(
            f"{label} matches {len(matches)} elements — "
            "disambiguate with --index <n> or --first before asserting on its state",
            hint="candidates: "
            + " | ".join(element_digest(el) for el in matches[:_MAX_CANDIDATES]),
        )
    if index is not None and index >= len(matches):
        return False, detail_tokens(
            "fail",
            sought=label,
            predicate="index",
            expected=index,
            actual=f"{len(matches)} matches",
        )
    el = matches[index] if index is not None else matches[0]
    if contains_all:
        contains_ok, contains_detail = check_contains_all(elements, el, contains_all)
        if not contains_ok:
            return False, detail_tokens(
                "fail", sought=label, id=el.id
            ) + " | " + contains_detail
    failures = self._check_predicates(el, self._node_state(xml, el), predicates)
    if failures:
        return False, detail_tokens("fail", sought=label, id=el.id) + " | " + "; ".join(
            failures
        )
    checks = ",".join(predicates) or "exists"
    if count is not None:
        checks += f",count={count}"
    return True, detail_tokens("pass", sought=label, id=el.id, checks=checks)


def expect(
    self: Engine,
    *,
    rid: str | None = None,
    text: str | None = None,
    desc: str | None = None,
    exists: bool = False,
    absent: bool = False,
    text_is: str | None = None,
    text_contains: str | None = None,
    checked: bool | None = None,
    enabled: bool | None = None,
    selected: bool | None = None,
    focused: bool | None = None,
    count: int | None = None,
    within: dict[str, Any] | None = None,
    same_parent_as: dict[str, Any] | None = None,
    contains_all: Sequence[dict[str, Any]] | None = None,
    index: int | None = None,
    first: bool = False,
    timeout_ms: int = 0,
    poll_ms: int = 250,
    observe: bool = False,
) -> ActionResult:
    """Assert something about the screen; ``ok=False`` means the assertion failed.

        This is the primitive that turns an acceptance-criteria list into a script: one
        criterion per call, exit code per criterion. ``timeout_ms`` polls until the
        assertion holds, which is what replaces a ``sleep`` guess — the flakiness the
        project's own testing guidance warns about.
        """
    selector = {
        "rid": normalize_selector_prefix("rid", rid),
        "text": normalize_selector_prefix("text", text),
        "desc": normalize_selector_prefix("desc", desc),
    }
    if len([v for v in selector.values() if v]) != 1:
        raise UsageError(
            "expect needs exactly one of --rid / --text / --desc",
            hint="e.g. `aua expect-and-analyze --rid notificationsButton --exists`",
        )
    if absent and exists:
        raise UsageError("--exists and --absent are mutually exclusive")
    if count is not None and count < 0:
        raise UsageError("--count must not be negative")
    if index is not None and index < 0:
        raise UsageError("--index must not be negative")
    if absent and any(value is not None for value in (within, same_parent_as, contains_all)):
        raise UsageError("--absent cannot be combined with structural predicates")
    if count == 0 and any(
        value is not None
        for value in (
            text_is,
            text_contains,
            checked,
            enabled,
            selected,
            focused,
            within,
            same_parent_as,
            contains_all,
        )
    ):
        raise UsageError("--count 0 cannot be combined with element state/structure predicates")
    normalized_within = (
        normalize_selector(within, field="within") if within is not None else None
    )
    normalized_same_parent = (
        normalize_selector(same_parent_as, field="same_parent_as")
        if same_parent_as is not None
        else None
    )
    normalized_contains = tuple(
        normalize_selector(value, field=f"contains_all[{position}]")
        for position, value in enumerate(contains_all or ())
    )
    if contains_all is not None and not normalized_contains:
        raise UsageError("contains_all must not be empty")
    predicates: dict[str, Any] = {}
    if absent:
        predicates["absent"] = True
    for name, value in (
        ("text_is", text_is),
        ("text_contains", text_contains),
        ("checked", checked),
        ("enabled", enabled),
        ("selected", selected),
        ("focused", focused),
    ):
        if value is not None:
            predicates[name] = value
    if not predicates or exists:
        predicates.setdefault("exists", True)
    # `--timeout` here polls until the assertion holds, which makes it an observation wait
    # wearing a different name — and the only one still unbounded, so `expect --timeout
    # 120000` was a way to block for two minutes without saying `wait`.
    timeout_ms, clamped_from, ceiling_ms = self._bounded_wait_ms(max(0, timeout_ms))
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        ok, detail = self._expect_once(
            selector,
            predicates,
            index=index,
            first=first,
            count=count,
            within=normalized_within,
            same_parent_as=normalized_same_parent,
            contains_all=normalized_contains,
        )
        if ok or time.monotonic() >= deadline:
            return self._say_the_wait_was_shortened(
                self._observe(
                    ActionResult(ok=ok, action="expect", detail=detail), observe, None
                ),
                clamped_from,
                ceiling_ms,
            )
        self._sleep_between_polls(max(50.0, float(poll_ms)), deadline)
