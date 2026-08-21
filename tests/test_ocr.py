"""Tests for OCR providers (PRD §7.2, §12, §13.1).

Test categories:
1. Unavailable providers report is_available().ok == False with a helpful reason.
2. Coordinate-conversion correctness via mocked engine calls (deterministic).
3. Real availability checks for apple_vision and rapidocr on this host.
4. Best-effort real OCR smoke test on a PIL image with rendered text.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.providers.ocr.apple_vision import AppleVisionOcrProvider
from android_ui_analyser.providers.ocr.easyocr import EasyOcrProvider
from android_ui_analyser.providers.ocr.paddleocr import PaddleOcrProvider
from android_ui_analyser.providers.ocr.rapidocr import RapidOcrProvider
from android_ui_analyser.providers.ocr.tesseract import TesseractOcrProvider

# ---------------------------------------------------------------------------
# Optional-extra gates
# ---------------------------------------------------------------------------
# apple_vision and rapidocr are optional extras, so `uv run pytest tests` on a plain
# checkout has neither. Tests that need one must skip, not fail: a red suite that means
# "you did not install an optional extra" trains everyone to ignore red, and it hid a real
# regression here once - a genuine failure was dismissed as this same noise.

_APPLE_OK = AppleVisionOcrProvider().is_available().ok
_RAPID_OK = RapidOcrProvider().is_available().ok

needs_apple = pytest.mark.skipif(_APPLE_OK is False, reason="extra not installed: [apple]")
needs_rapid = pytest.mark.skipif(_RAPID_OK is False, reason="extra not installed: rapidocr")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screen_image(width: int = 300, height: int = 100, text: str = "") -> ScreenImage:
    """Return a ScreenImage with optional rendered text."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    if text:
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ScreenImage(buf.getvalue(), width=width, height=height)


# ---------------------------------------------------------------------------
# 1. Unavailable providers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _import_blocked(*names: str) -> Iterator[None]:
    """Make ``import <name>`` raise ImportError, whatever this host happens to have installed.

    A ``None`` entry in ``sys.modules`` is the documented way to make an import fail, and it is
    what makes these tests state a fact about the code instead of a fact about the machine.
    """
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in names}
    try:
        for name in names:
            sys.modules[name] = None  # type: ignore[assignment]
        yield
    finally:
        for name, module in saved.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]


class TestUnavailableProviders:
    """A missing dependency must be *reported*, with a reason naming the install extra.

    Absence is simulated rather than assumed. These tests used to read the host: they asserted
    ``ok is False`` against whatever was really installed, so they passed on a plain checkout and
    failed the moment someone installed the extra — the exact inversion of the rule stated at the
    top of this file. A suite that goes red because you installed something trains everyone to
    ignore red just as effectively as one that goes red because you did not.
    """

    def test_paddleocr_unavailable(self) -> None:
        with _import_blocked("paddleocr"):
            avail = PaddleOcrProvider().is_available()
        assert avail.ok is False
        # The reason must mention the pip install extra
        assert "paddle" in avail.reason.lower() or "paddleocr" in avail.reason.lower()

    def test_tesseract_unavailable(self) -> None:
        with _import_blocked("pytesseract"):
            avail = TesseractOcrProvider().is_available()
        assert avail.ok is False
        assert "tesseract" in avail.reason.lower()

    def test_easyocr_unavailable(self) -> None:
        with _import_blocked("easyocr"):
            avail = EasyOcrProvider().is_available()
        assert avail.ok is False
        assert "easyocr" in avail.reason.lower()

    def test_unavailable_recognize_returns_empty_list(self) -> None:
        """recognize() must return [] (never raise) when the dep is missing."""
        image = _make_screen_image()
        with _import_blocked("paddleocr", "pytesseract", "easyocr"):
            assert PaddleOcrProvider().recognize(image) == []
            assert TesseractOcrProvider().recognize(image) == []
            assert EasyOcrProvider().recognize(image) == []


# ---------------------------------------------------------------------------
# 2. Coordinate-conversion tests (mocked engine calls)
# ---------------------------------------------------------------------------


def _make_vision_mocks(observations: list, fast_level_value: int = 1) -> tuple:
    """Return (mock_vision_module, mock_quartz_module, mock_req) pre-wired for testing."""
    mock_quartz = MagicMock()
    mock_quartz.CFDataCreate.return_value = b"fake_data"

    mock_handler = MagicMock()
    mock_handler.performRequests_error_.return_value = (True, None)

    mock_req = MagicMock()
    mock_req.results.return_value = observations

    mock_vision = MagicMock()
    mock_vision.VNImageRequestHandler.alloc.return_value.initWithData_options_.return_value = (
        mock_handler
    )
    mock_vision.VNRecognizeTextRequest.alloc.return_value.init.return_value = mock_req
    mock_vision.VNRequestTextRecognitionLevelAccurate = 0
    mock_vision.VNRequestTextRecognitionLevelFast = fast_level_value

    return mock_vision, mock_quartz, mock_req


def _make_fake_obs(
    text: str,
    conf: float,
    norm_x: float,
    norm_y: float,
    norm_w: float,
    norm_h: float,
) -> MagicMock:
    """Build a fake VNRecognizedTextObservation-like mock."""
    candidate = MagicMock()
    candidate.string.return_value = text

    origin = MagicMock()
    origin.x = norm_x
    origin.y = norm_y
    size = MagicMock()
    size.width = norm_w
    size.height = norm_h
    bbox = MagicMock()
    bbox.origin = origin
    bbox.size = size

    obs = MagicMock()
    obs.topCandidates_.return_value = [candidate]
    obs.confidence.return_value = conf
    obs.boundingBox.return_value = bbox
    return obs


@needs_apple
class TestAppleVisionCoordConversion:
    """Monkeypatch the Vision handler to return a known raw observation and
    assert that recognize() converts the bounding box correctly.

    Guarded like the rest of the apple_vision tests: mocking `Vision`/`Quartz` in `sys.modules`
    is not enough on a host without pyobjc, because `recognize()` checks `is_available()` first
    and returns an empty list before any mock is consulted. Unguarded, these three were the only
    thing standing between this suite and a green Linux CI job.

    Since Vision/Quartz are lazy-imported inside the methods, we patch them
    in sys.modules so the `import` statements inside the provider pick up our
    mocks.
    """

    def test_coordinate_conversion(self) -> None:
        """
        Given a normalized bbox at (0.1, 0.6, 0.5, 0.2) (x, y, w, h; bottom-left origin)
        on a 200x100 image, expect pixel bounds (20, 20, 120, 40) (top-left origin).

        Derivation:
          x1 = 0.1 * 200 = 20
          y1 = (1 - 0.6 - 0.2) * 100 = 0.2 * 100 = 20
          x2 = (0.1 + 0.5) * 200 = 120
          y2 = (1 - 0.6) * 100 = 40
        """
        image = _make_screen_image(width=200, height=100)
        obs = _make_fake_obs("Test", 0.9, norm_x=0.1, norm_y=0.6, norm_w=0.5, norm_h=0.2)
        mock_vision, mock_quartz, _mock_req = _make_vision_mocks([obs])

        provider = AppleVisionOcrProvider()

        with patch.dict(sys.modules, {"Vision": mock_vision, "Quartz": mock_quartz}):
            result = provider.recognize(image)

        assert len(result) == 1
        tb = result[0]
        assert tb.text == "Test"
        assert tb.bounds == (20, 20, 120, 40)
        assert abs(tb.confidence - 0.9) < 1e-6

    def test_empty_observations_returns_empty_list(self) -> None:
        """If Vision returns no observations, recognize() returns []."""
        image = _make_screen_image(width=200, height=100)
        mock_vision, mock_quartz, _mock_req = _make_vision_mocks([])
        provider = AppleVisionOcrProvider()

        with patch.dict(sys.modules, {"Vision": mock_vision, "Quartz": mock_quartz}):
            result = provider.recognize(image)

        assert result == []

    def test_downscale_maps_boxes_back_to_original_screen(self) -> None:
        image = _make_screen_image(width=1080, height=2400)
        obs = _make_fake_obs("Test", 0.9, norm_x=0.1, norm_y=0.6, norm_w=0.5, norm_h=0.2)
        mock_vision, mock_quartz, _mock_req = _make_vision_mocks([obs])
        provider = AppleVisionOcrProvider(settings={"max_width": 720})

        with patch.dict(sys.modules, {"Vision": mock_vision, "Quartz": mock_quartz}):
            result = provider.recognize(image)

        encoded = mock_quartz.CFDataCreate.call_args.args[1]
        assert Image.open(io.BytesIO(encoded)).size == (720, 1600)
        assert result[0].bounds == (108, 480, 648, 960)

    def test_recognition_level_fast_setting(self) -> None:
        """settings['recognition_level'] = 'fast' sets the correct Vision constant."""
        image = _make_screen_image(width=100, height=50)
        provider = AppleVisionOcrProvider(settings={"recognition_level": "fast"})
        mock_vision, mock_quartz, mock_req = _make_vision_mocks([], fast_level_value=1)

        with patch.dict(sys.modules, {"Vision": mock_vision, "Quartz": mock_quartz}):
            provider.recognize(image)

        # Verify setRecognitionLevel_ was called with the FAST constant (1)
        mock_req.setRecognitionLevel_.assert_called_once_with(1)


@needs_rapid
class TestRapidOcrCoordConversion:
    """Monkeypatch the cached _engine on RapidOcrProvider to return a known raw
    result, then assert that recognize() converts the 4-point polygon correctly.

    We inject the mock engine directly (provider._engine = mock_callable) to avoid
    sys.modules patching side-effects that can arise when rapidocr_onnxruntime's
    cv2 transitive dependency interacts with numpy's C-extension loading.
    """

    def _make_provider_with_engine(self, fake_result: tuple) -> RapidOcrProvider:
        """Return a RapidOcrProvider whose _engine is wired to return fake_result."""
        provider = RapidOcrProvider()
        mock_engine = MagicMock(return_value=fake_result)
        provider._engine = mock_engine
        return provider

    def test_coordinate_conversion(self) -> None:
        """
        Given a 4-point polygon [[10,20],[80,20],[80,50],[10,50]] with text 'Hello' and score 0.95,
        expect TextBox(text='Hello', bounds=(10,20,80,50), confidence=0.95).
        """
        image = _make_screen_image(width=200, height=100)
        fake_result = (
            [
                [
                    [[10.0, 20.0], [80.0, 20.0], [80.0, 50.0], [10.0, 50.0]],
                    "Hello",
                    0.95,
                ]
            ],
            [0.1, 0.0, 0.05],
        )
        provider = self._make_provider_with_engine(fake_result)
        result = provider.recognize(image)

        assert len(result) == 1
        tb = result[0]
        assert tb.text == "Hello"
        assert tb.bounds == (10, 20, 80, 50)
        assert abs(tb.confidence - 0.95) < 1e-6

    def test_none_result_returns_empty_list(self) -> None:
        """RapidOCR returns (None, None) on blank images; recognize() must return []."""
        image = _make_screen_image(width=100, height=50)
        provider = self._make_provider_with_engine((None, None))
        result = provider.recognize(image)
        assert result == []

    def test_multiple_boxes(self) -> None:
        """Multiple boxes are each converted correctly."""
        image = _make_screen_image(width=300, height=200)
        fake_result = (
            [
                [[[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]], "foo", 0.8],
                [[[60.0, 10.0], [120.0, 10.0], [120.0, 30.0], [60.0, 30.0]], "bar", 0.7],
            ],
            [0.05, 0.0, 0.02],
        )
        provider = self._make_provider_with_engine(fake_result)
        result = provider.recognize(image)

        assert len(result) == 2
        assert result[0].text == "foo"
        assert result[0].bounds == (0, 0, 50, 20)
        assert result[1].text == "bar"
        assert result[1].bounds == (60, 10, 120, 30)


# ---------------------------------------------------------------------------
# 3. Real availability checks
# ---------------------------------------------------------------------------


class TestRealAvailability:
    """On this macOS/arm64 host with pyobjc Vision and rapidocr-onnxruntime installed,
    both providers must report available."""

    @needs_apple
    def test_apple_vision_available(self) -> None:
        provider = AppleVisionOcrProvider()
        avail = provider.is_available()
        assert avail.ok is True, f"Expected apple_vision available, got: {avail.reason}"

    @needs_rapid
    def test_rapidocr_available(self) -> None:
        provider = RapidOcrProvider()
        avail = provider.is_available()
        assert avail.ok is True, f"Expected rapidocr available, got: {avail.reason}"


# ---------------------------------------------------------------------------
# 4. Best-effort real OCR smoke tests
# ---------------------------------------------------------------------------


class TestRealOcrSmoke:
    """Run real OCR on a synthetic image with rendered text.

    These tests are lenient: we guard with is_available() and only assert that
    the return value is a list.  Coordinate-correctness is covered by the mocked
    tests above.
    """

    def test_apple_vision_smoke(self) -> None:
        provider = AppleVisionOcrProvider()
        if not provider.is_available().ok:
            pytest.skip("apple_vision not available")
        image = _make_screen_image(width=300, height=100, text="Hello")
        result = provider.recognize(image)
        assert isinstance(result, list)
        # On this host, Vision should detect text in a clean synthetic image.
        assert len(result) >= 1
        texts = [tb.text for tb in result]
        assert any("Hello" in t for t in texts), f"Expected 'Hello' in {texts}"

    def test_rapidocr_smoke(self) -> None:
        provider = RapidOcrProvider()
        if not provider.is_available().ok:
            pytest.skip("rapidocr not available")
        image = _make_screen_image(width=300, height=100, text="Hello")
        result = provider.recognize(image)
        assert isinstance(result, list)

    def test_apple_vision_blank_returns_list(self) -> None:
        """A blank white image must return [] without raising."""
        provider = AppleVisionOcrProvider()
        if not provider.is_available().ok:
            pytest.skip("apple_vision not available")
        image = _make_screen_image(width=200, height=100, text="")
        result = provider.recognize(image)
        assert isinstance(result, list)

    def test_rapidocr_blank_returns_list(self) -> None:
        """A blank white image must return [] without raising."""
        provider = RapidOcrProvider()
        if not provider.is_available().ok:
            pytest.skip("rapidocr not available")
        image = _make_screen_image(width=200, height=100, text="")
        result = provider.recognize(image)
        assert isinstance(result, list)
