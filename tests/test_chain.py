"""AC4 — fallback-chain runner — and the open/closed "provider selectable by config alone".

Covers PRD §7 (chain runner: skip unavailable, advance on error/empty, raise
ProviderError when exhausted → exit 4) and STEP 2 (a new provider is usable by editing
config only, with zero engine/CLI changes).
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import ExitCode, ProviderError
from android_ui_analyser.providers.base import Availability, OcrProvider, ScreenImage, TextBox
from android_ui_analyser.providers.registry import ProviderFactory, register_ocr, run_chain
from conftest import StubOcr, make_chain, make_config, make_screen_image

IMG = make_screen_image()
OK = [TextBox(text="hi", bounds=(0, 0, 5, 5), confidence=0.9)]


def _op(p):
    return p.recognize(IMG)


def test_fail_then_ok_skips_failer_returns_second() -> None:
    failer = StubOcr(raises=RuntimeError("boom"))
    ok = StubOcr(result=OK)
    result, _name = run_chain(make_chain("ocr", [failer, ok]), _op)
    assert result == OK
    assert failer.calls == 1 and ok.calls == 1  # failer was tried, then ok


def test_fail_then_fail_raises_provider_error_exit_4() -> None:
    chain = make_chain("ocr", [StubOcr(raises=RuntimeError("a")), StubOcr(raises=ValueError("b"))])
    with pytest.raises(ProviderError) as ei:
        run_chain(chain, _op)
    assert ei.value.exit_code == ExitCode.PROVIDER == 4
    # attempts are recorded for a useful message
    assert len(ei.value.attempts) == 2


def test_unavailable_provider_is_skipped() -> None:
    unavail = StubOcr(available=False, reason="missing dep")
    ok = StubOcr(result=OK)
    result, _ = run_chain(make_chain("ocr", [unavail, ok]), _op)
    assert result == OK
    assert unavail.calls == 0  # never invoked when unavailable


def test_empty_result_advances_to_next() -> None:
    empty = StubOcr(result=[])
    ok = StubOcr(result=OK)
    result, _ = run_chain(make_chain("ocr", [empty, ok]), _op)
    assert result == OK


def test_all_empty_returns_empty_not_error() -> None:
    # A genuinely empty screen must not raise — a clean-but-empty result is returned.
    result, _ = run_chain(make_chain("ocr", [StubOcr(result=[]), StubOcr(result=[])]), _op)
    assert result == []


def test_all_unavailable_raises() -> None:
    chain = make_chain(
        "ocr", [StubOcr(available=False, reason="x"), StubOcr(available=False, reason="y")]
    )
    with pytest.raises(ProviderError):
        run_chain(chain, _op)


# --------------------------------------------------------------- open/closed (STEP 2)


def test_dummy_provider_selectable_by_config_alone() -> None:
    """The conftest-registered ``dummy`` OCR provider is selected purely via config."""
    cfg = make_config(ocr={"enabled": True, "chain": ["dummy"]})
    chain = ProviderFactory(cfg).build_chain("ocr")
    assert [p.name for p in chain.providers] == ["dummy"]
    assert chain.providers[0].recognize(IMG)  # the dummy returns a TextBox


def test_brand_new_provider_registered_then_selected() -> None:
    """Register a provider at runtime + select it by config — no engine/CLI edits."""

    @register_ocr("brand_new_xyz")
    class _New(OcrProvider):
        def is_available(self) -> Availability:
            return Availability(True, "ok")

        def recognize(self, image: ScreenImage) -> list[TextBox]:
            return [TextBox(text="new", bounds=(0, 0, 1, 1))]

    cfg = make_config(ocr={"chain": ["brand_new_xyz"]})
    chain = ProviderFactory(cfg).build_chain("ocr")
    assert [p.name for p in chain.providers] == ["brand_new_xyz"]
    result, name = run_chain(chain, _op)
    assert name == "brand_new_xyz"
    assert result[0].text == "new"


def test_unknown_provider_in_chain_is_skipped_not_fatal() -> None:
    cfg = make_config(ocr={"chain": ["does_not_exist", "dummy"]})
    chain = ProviderFactory(cfg).build_chain("ocr")
    assert [p.name for p in chain.providers] == ["dummy"]  # bogus name skipped
