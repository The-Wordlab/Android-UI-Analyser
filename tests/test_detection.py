"""Tests for YOLO and OmniParser detection providers (PRD §13 AC / task spec).

All tests are deterministic and require no torch/ultralytics installation.
Heavy-dep paths are exercised by monkeypatching the provider's internal model
loader so the test never touches torch.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from android_ui_analyser.providers.base import Availability, DetBox
from android_ui_analyser.providers.detection.omniparser import OmniParserProvider
from android_ui_analyser.providers.detection.yolo import YoloProvider
from conftest import make_screen_image

# --------------------------------------------------------------------------- helpers


def _fake_result(boxes_data: list[dict[str, Any]]) -> list[Any]:
    """Build a fake ultralytics Results list from a list of box dicts.

    Each dict may have: xyxy (list[float]), conf (float), cls (int), names (dict).
    """
    results = []
    for box_spec in boxes_data:
        result = MagicMock()
        box = MagicMock()

        import numpy as np

        xyxy = box_spec.get("xyxy", [0.0, 0.0, 10.0, 10.0])
        box.xyxy = [np.array(xyxy, dtype=float)]
        box.conf = [box_spec.get("conf", 0.9)]
        box.cls = [box_spec.get("cls", 0)]

        result.boxes = [box]
        result.names = box_spec.get("names", {0: "icon"})
        results.append(result)
    return results


# =========================================================================== yolo


class TestYoloAvailability:
    def test_no_weights_returns_false(self):
        provider = YoloProvider(settings={})
        avail = provider.is_available()
        assert isinstance(avail, Availability)
        assert avail.ok is False
        assert "weights" in avail.reason.lower()

    def test_none_weights_returns_false(self):
        provider = YoloProvider(settings={"weights": None})
        avail = provider.is_available()
        assert avail.ok is False
        assert "weights" in avail.reason.lower()

    def test_nonexistent_path_returns_false(self, tmp_path):
        missing = tmp_path / "nonexistent.pt"
        provider = YoloProvider(settings={"weights": str(missing)})
        avail = provider.is_available()
        assert avail.ok is False
        # Reason should mention either the weights path issue or the install extra.
        assert ("weights" in avail.reason.lower()) or ("yolo" in avail.reason.lower())

    def test_existing_path_but_no_deps_returns_false(self, tmp_path, monkeypatch):
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"fake")

        # Simulate ultralytics not importable.
        import android_ui_analyser.providers.detection.yolo as yolo_mod

        original = yolo_mod._check_heavy_deps

        def _fake_no_deps():
            return (
                False,
                "ultralytics and/or torch not installed; run: pip install android-ui-analyser[yolo]",
            )

        monkeypatch.setattr(yolo_mod, "_check_heavy_deps", _fake_no_deps)
        try:
            provider = YoloProvider(settings={"weights": str(weights)})
            avail = provider.is_available()
            assert avail.ok is False
            assert "yolo" in avail.reason.lower()
        finally:
            monkeypatch.setattr(yolo_mod, "_check_heavy_deps", original)

    def test_existing_path_and_deps_returns_true(self, tmp_path, monkeypatch):
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"fake")

        import android_ui_analyser.providers.detection.yolo as yolo_mod

        monkeypatch.setattr(yolo_mod, "_check_heavy_deps", lambda: (True, ""))
        provider = YoloProvider(settings={"weights": str(weights)})
        avail = provider.is_available()
        assert avail.ok is True


class TestYoloDetectMapping:
    """Monkeypatch _load_model to inject a fake ultralytics model."""

    def _make_provider_with_fake_model(self, tmp_path, monkeypatch, boxes_data):
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"fake")

        fake_results = _fake_result(boxes_data)
        fake_model = MagicMock()
        fake_model.return_value = fake_results

        provider = YoloProvider(settings={"weights": str(weights), "device": "cpu", "conf": 0.1})
        monkeypatch.setattr(provider, "_load_model", lambda path, dev: fake_model)
        return provider

    def test_maps_boxes_to_detbox(self, tmp_path, monkeypatch):
        boxes_data = [
            {"xyxy": [10.0, 20.0, 110.0, 220.0], "conf": 0.8, "cls": 0, "names": {0: "button"}},
        ]
        provider = self._make_provider_with_fake_model(tmp_path, monkeypatch, boxes_data)
        image = make_screen_image(200, 400)
        results = provider.detect(image)

        assert len(results) == 1
        box = results[0]
        assert isinstance(box, DetBox)
        assert box.bounds == (10, 20, 110, 220)
        assert box.label == "button"
        assert box.confidence == pytest.approx(0.8)
        assert box.interactable is True

    def test_filters_by_conf_threshold(self, tmp_path, monkeypatch):
        boxes_data = [
            {"xyxy": [0.0, 0.0, 50.0, 50.0], "conf": 0.05, "cls": 0, "names": {0: "icon"}},
            {
                "xyxy": [0.0, 0.0, 50.0, 50.0],
                "conf": 0.9,
                "cls": 1,
                "names": {0: "icon", 1: "button"},
            },
        ]
        # conf threshold = 0.1 → first box dropped
        provider = self._make_provider_with_fake_model(tmp_path, monkeypatch, boxes_data)
        image = make_screen_image(100, 100)
        results = provider.detect(image)

        assert len(results) == 1
        assert results[0].label == "button"

    def test_multiple_boxes_mapped(self, tmp_path, monkeypatch):
        boxes_data = [
            {"xyxy": [0.0, 0.0, 10.0, 10.0], "conf": 0.7, "cls": 0, "names": {0: "a"}},
            {"xyxy": [20.0, 20.0, 30.0, 30.0], "conf": 0.6, "cls": 1, "names": {0: "a", 1: "b"}},
        ]
        provider = self._make_provider_with_fake_model(tmp_path, monkeypatch, boxes_data)
        image = make_screen_image(100, 100)
        results = provider.detect(image)

        assert len(results) == 2
        assert results[0].bounds == (0, 0, 10, 10)
        assert results[1].bounds == (20, 20, 30, 30)

    def test_returns_empty_list_on_model_error(self, tmp_path, monkeypatch):
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"fake")
        provider = YoloProvider(settings={"weights": str(weights)})

        def _bad_loader(path, dev):
            raise RuntimeError("model load failed")

        monkeypatch.setattr(provider, "_load_model", _bad_loader)
        image = make_screen_image()
        results = provider.detect(image)
        assert results == []


# =========================================================================== omniparser


class TestOmniParserAvailability:
    def test_no_accept_agpl_returns_false(self):
        provider = OmniParserProvider(settings={})
        avail = provider.is_available()
        assert avail.ok is False
        assert "agpl" in avail.reason.lower()

    def test_accept_agpl_false_returns_false(self):
        provider = OmniParserProvider(settings={"accept_agpl": False})
        avail = provider.is_available()
        assert avail.ok is False
        assert "agpl" in avail.reason.lower()

    def test_accept_agpl_true_but_deps_absent_returns_false(self, monkeypatch):
        import android_ui_analyser.providers.detection.omniparser as omni_mod

        monkeypatch.setattr(
            omni_mod,
            "_check_heavy_deps",
            lambda: (
                False,
                "ultralytics not installed; run: pip install android-ui-analyser[omniparser]",
            ),
        )
        provider = OmniParserProvider(settings={"accept_agpl": True})
        avail = provider.is_available()
        assert avail.ok is False
        assert "omniparser" in avail.reason.lower()

    def test_accept_agpl_true_and_deps_present_returns_true(self, monkeypatch):
        import android_ui_analyser.providers.detection.omniparser as omni_mod

        monkeypatch.setattr(omni_mod, "_check_heavy_deps", lambda: (True, ""))
        provider = OmniParserProvider(settings={"accept_agpl": True})
        avail = provider.is_available()
        assert avail.ok is True


class TestOmniParserAgplWarning:
    """Verify the one-time AGPL warning fires on first detect() and maps boxes correctly."""

    def _make_provider_with_fake_model(self, monkeypatch, boxes_data):
        """Return a provider whose _load_model is patched to not touch torch."""
        import android_ui_analyser.providers.detection.omniparser as omni_mod

        # Reset the module-level warning flag so this test starts clean.
        monkeypatch.setattr(omni_mod, "_agpl_warned", False)

        fake_results = _fake_result(boxes_data)
        fake_model = MagicMock()
        fake_model.return_value = fake_results

        provider = OmniParserProvider(
            settings={"accept_agpl": True, "device": "cpu", "box_threshold": 0.01}
        )
        monkeypatch.setattr(provider, "_load_model", lambda: fake_model)
        return provider

    def test_agpl_warning_fires_on_first_detect(self, monkeypatch, caplog):
        boxes_data = [
            {"xyxy": [5.0, 5.0, 50.0, 50.0], "conf": 0.7, "cls": 0, "names": {0: "icon"}},
        ]
        provider = self._make_provider_with_fake_model(monkeypatch, boxes_data)
        image = make_screen_image()

        with caplog.at_level(logging.WARNING, logger="android_ui_analyser.providers"):
            provider.detect(image)

        warning_msgs = [r.message for r in caplog.records if "agpl" in r.message.lower()]
        assert len(warning_msgs) >= 1, "Expected at least one AGPL warning log"

    def test_agpl_warning_fires_only_once(self, monkeypatch, caplog):
        boxes_data = [
            {"xyxy": [5.0, 5.0, 50.0, 50.0], "conf": 0.7, "cls": 0, "names": {0: "icon"}},
        ]
        provider = self._make_provider_with_fake_model(monkeypatch, boxes_data)
        image = make_screen_image()

        with caplog.at_level(logging.WARNING, logger="android_ui_analyser.providers"):
            provider.detect(image)
            provider.detect(image)

        warning_msgs = [r.message for r in caplog.records if "agpl" in r.message.lower()]
        assert len(warning_msgs) == 1, "AGPL warning must fire exactly once per process"

    def test_detect_maps_boxes_correctly(self, monkeypatch):
        boxes_data = [
            {"xyxy": [10.0, 20.0, 110.0, 220.0], "conf": 0.65, "cls": 0, "names": {0: "icon"}},
        ]
        provider = self._make_provider_with_fake_model(monkeypatch, boxes_data)
        image = make_screen_image(200, 400)
        results = provider.detect(image)

        assert len(results) == 1
        box = results[0]
        assert isinstance(box, DetBox)
        assert box.bounds == (10, 20, 110, 220)
        assert box.label == "icon"
        assert box.confidence == pytest.approx(0.65)
        assert box.interactable is True

    def test_detect_returns_empty_on_model_error(self, monkeypatch):
        import android_ui_analyser.providers.detection.omniparser as omni_mod

        monkeypatch.setattr(omni_mod, "_agpl_warned", False)

        provider = OmniParserProvider(settings={"accept_agpl": True})

        def _bad_loader():
            raise RuntimeError("cannot load model")

        monkeypatch.setattr(provider, "_load_model", _bad_loader)
        image = make_screen_image()
        results = provider.detect(image)
        assert results == []


# =========================================================================== import safety


def test_yolo_module_does_not_import_torch_at_module_level():
    """yolo.py must not drag torch/ultralytics in at import time."""
    import importlib
    import sys

    # Remove any previously imported version so we get a fresh import.
    for mod in list(sys.modules.keys()):
        if mod.startswith("android_ui_analyser.providers.detection.yolo"):
            del sys.modules[mod]

    before = set(sys.modules.keys())
    importlib.import_module("android_ui_analyser.providers.detection.yolo")
    after = set(sys.modules.keys())
    new_mods = after - before
    heavy = {m for m in new_mods if m.startswith("torch") or m.startswith("ultralytics")}
    assert not heavy, f"yolo.py imported heavy deps at module level: {heavy}"


def test_omniparser_module_does_not_import_torch_at_module_level():
    """omniparser.py must not drag torch/ultralytics/huggingface_hub in at import time."""
    import importlib
    import sys

    for mod in list(sys.modules.keys()):
        if mod.startswith("android_ui_analyser.providers.detection.omniparser"):
            del sys.modules[mod]

    before = set(sys.modules.keys())
    importlib.import_module("android_ui_analyser.providers.detection.omniparser")
    after = set(sys.modules.keys())
    new_mods = after - before
    heavy = {
        m
        for m in new_mods
        if m.startswith("torch") or m.startswith("ultralytics") or m.startswith("huggingface_hub")
    }
    assert not heavy, f"omniparser.py imported heavy deps at module level: {heavy}"
