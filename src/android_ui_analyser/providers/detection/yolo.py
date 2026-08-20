"""YOLO detection provider (PRD §7.2, detection/yolo).

Uses user-supplied Ultralytics YOLO weights to detect UI elements.  The
weights path must be configured explicitly; this provider is deliberately
license-clean once the user provides their own checkpoint (e.g. fine-tuned
on RICO/VINS).  No weights are bundled.

Heavy deps (ultralytics, torch) are lazy-imported so the core CLI stays
importable without them.

Usage in config.yaml::

    detection:
      chain: [yolo]
    models:
      yolo:
        weights: ~/models/ui-yolo.pt
        device: mps       # mps | cuda | cpu
        conf: 0.25
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import Availability, DetBox, DetectionProvider, ScreenImage
from ..registry import register_detection


def _check_heavy_deps() -> tuple[bool, str]:
    """Return (importable, reason).  Never raises."""
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401

        return True, ""
    except ImportError:
        return (
            False,
            "ultralytics and/or torch not installed; run: pip install android-ui-analyser[yolo]",
        )


@register_detection("yolo")
class YoloProvider(DetectionProvider):
    """Generic Ultralytics YOLO provider driven by user-supplied weights.

    Settings (``models.yolo`` block in config):
    - ``weights``: path to a ``.pt`` file (required; may use ``~``).
    - ``device``: ``"mps"`` | ``"cuda"`` | ``"cpu"`` (default ``"cpu"``).
    - ``conf``: confidence threshold (default ``0.25``).
    """

    # Cache loaded model instances keyed by (resolved_weights_path, device).
    _model_cache: dict[tuple[str, str], Any]

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        super().__init__(settings)
        self._model_cache = {}

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> Availability:
        """Cheap check — no network, no model load."""
        weights_raw = self.settings.get("weights")
        if not weights_raw:
            return Availability(
                False,
                "no YOLO weights configured (set models.yolo.weights to a .pt path)",
            )

        weights_path = Path(os.path.expanduser(str(weights_raw)))
        if not weights_path.exists():
            return Availability(
                False,
                "no YOLO weights configured (set models.yolo.weights to a .pt path)",
            )

        deps_ok, deps_reason = _check_heavy_deps()
        if not deps_ok:
            return Availability(False, deps_reason)

        return Availability(True, "yolo ready")

    # ------------------------------------------------------------------
    # Model loading (overridable for tests)
    # ------------------------------------------------------------------

    def _load_model(self, weights_path: str, device: str) -> Any:
        """Load and return a YOLO model.  Cached by (weights_path, device)."""
        cache_key = (weights_path, device)
        if cache_key not in self._model_cache:
            from ultralytics import YOLO  # lazy import

            model = YOLO(weights_path)
            self._model_cache[cache_key] = model
        return self._model_cache[cache_key]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, image: ScreenImage) -> list[DetBox]:
        """Run YOLO inference and return DetBox list.  Returns [] on any error."""
        try:
            return self._detect_inner(image)
        except Exception:
            return []

    def _detect_inner(self, image: ScreenImage) -> list[DetBox]:
        weights_raw = self.settings.get("weights", "")
        weights_path = str(Path(os.path.expanduser(str(weights_raw))).resolve())
        device: str = self.settings.get("device", "cpu")
        conf_threshold: float = float(self.settings.get("conf", 0.25))

        model = self._load_model(weights_path, device)

        # Prefer a numpy RGB array; fall back to saving a temp PNG.
        try:
            rgb_array = image.numpy()
            results = model(rgb_array, device=device, verbose=False)
        except Exception:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                image.save(tmp_path)
                results = model(tmp_path, device=device, verbose=False)
            finally:
                import contextlib

                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

        boxes: list[DetBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0]) if box.conf is not None else None
                if conf is not None and conf < conf_threshold:
                    continue
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                cls_id = int(box.cls[0]) if box.cls is not None else None
                label: str | None = None
                if cls_id is not None and result.names:
                    label = result.names.get(cls_id)
                boxes.append(
                    DetBox(
                        bounds=(x1, y1, x2, y2),
                        label=label,
                        interactable=True,
                        confidence=conf,
                    )
                )
        return boxes
