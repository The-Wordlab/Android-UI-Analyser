"""OmniParser v2 detection-only provider (PRD §7.2, §14, §19).

Uses the YOLOv8 ``icon_detect`` model from ``microsoft/OmniParser-v2.0`` to
locate UI icons/elements.  The Florence-2 caption model is intentionally
skipped for speed and to avoid pulling in a second large dependency.

AGPL GATE
---------
The ``icon_detect`` weights are released under AGPL-3.0.  This provider will
refuse to run unless the caller explicitly opts in by setting::

    models:
      omniparser:
        accept_agpl: true

Without that flag the provider reports itself as unavailable with a clear
message.  Do NOT set ``accept_agpl: true`` in a commercially-shipped product
without reviewing your AGPL obligations.

SECURITY NOTE (CVE-2025-55322)
-------------------------------
CVE-2025-55322 affects the OmniTool *controller server* included in
OmniParser pre-2.0.1.  This provider uses **only** the YOLO weights — it
never launches or imports the OmniTool server component.  The huggingface_hub
download targets ``microsoft/OmniParser-v2.0`` which corresponds to the >=2.0.1
patched release; callers should treat any local cache from a pre-2.0.1 checkout
as potentially affected and re-download.

Heavy deps (ultralytics, torch, huggingface_hub) are lazy-imported so the
core CLI stays importable without them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..base import Availability, DetBox, DetectionProvider, ScreenImage
from ..registry import register_detection

_LOGGER = logging.getLogger("android_ui_analyser.providers")

# Module-level flag so the AGPL warning fires at most once per process.
_agpl_warned: bool = False

# HuggingFace repo and relative path for the icon_detect weights.
_HF_REPO = "microsoft/OmniParser-v2.0"
_HF_FILENAME = "icon_detect/model.pt"

# Default cache directory (resolved at call time so tests can override via env).
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "android-ui-analyser" / "omniparser"


def _check_heavy_deps() -> tuple[bool, str]:
    """Return (importable, reason).  Never raises."""
    missing = []
    for mod in ("ultralytics", "torch", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        pkgs = " ".join(missing)
        return (
            False,
            f"{pkgs} not installed; run: pip install android-ui-analyser[omniparser]",
        )
    return True, ""


@register_detection("omniparser")
class OmniParserProvider(DetectionProvider):
    """OmniParser v2 icon detection via YOLOv8.

    Settings (``models.omniparser`` block in config):
    - ``accept_agpl`` (bool, **required**): must be ``true`` to enable; see module docstring.
    - ``device``: ``"mps"`` | ``"cuda"`` | ``"cpu"`` (default ``"cpu"``).
    - ``box_threshold`` (float): confidence threshold (default ``0.05``).
    - ``cache_dir`` (str | None): override weight cache location.

    Security note: we use only the YOLO weights, never the OmniTool controller
    server that carried CVE-2025-55322 (patched in >=2.0.1).
    """

    _model: Any  # loaded on first use

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        super().__init__(settings)
        self._model = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> Availability:
        """Cheap check — no network, no model load."""
        if self.settings.get("accept_agpl") is not True:
            return Availability(
                False,
                (
                    "OmniParser icon_detect is AGPL-3.0; "
                    "set models.omniparser.accept_agpl: true to enable "
                    "(not for commercial shipping)"
                ),
            )

        deps_ok, deps_reason = _check_heavy_deps()
        if not deps_ok:
            return Availability(False, deps_reason)

        return Availability(True, "omniparser ready")

    # ------------------------------------------------------------------
    # Weight download / model loading (overridable for tests)
    # ------------------------------------------------------------------

    def _get_weights_path(self) -> str:
        """Download icon_detect weights from HuggingFace if not cached."""
        from huggingface_hub import hf_hub_download  # lazy import

        cache_dir_raw = self.settings.get("cache_dir", str(_DEFAULT_CACHE_DIR))
        cache_dir = Path(os.path.expanduser(str(cache_dir_raw)))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # hf_hub_download places the file in a content-addressed subfolder under
        # cache_dir; we pass cache_dir so it respects our location preference.
        weights_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_HF_FILENAME,
            cache_dir=str(cache_dir),
        )
        return weights_path

    def _load_model(self) -> Any:
        """Load and return a YOLO model.  Cached on the instance."""
        if self._model is None:
            from ultralytics import YOLO  # lazy import

            weights_path = self._get_weights_path()
            self._model = YOLO(weights_path)
        return self._model

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, image: ScreenImage) -> list[DetBox]:
        """Run OmniParser icon_detect and return DetBox list.  Returns [] on error."""
        try:
            return self._detect_inner(image)
        except Exception:
            return []

    def _detect_inner(self, image: ScreenImage) -> list[DetBox]:
        global _agpl_warned  # noqa: PLW0603

        device: str = self.settings.get("device", "cpu")
        box_threshold: float = float(self.settings.get("box_threshold", 0.05))

        model = self._load_model()

        if not _agpl_warned:
            _LOGGER.warning(
                "OmniParser icon_detect weights are AGPL-3.0. "
                "Ensure your usage complies with AGPL before shipping commercially. "
                "(This message is logged once per process.)"
            )
            _agpl_warned = True

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
                if conf is not None and conf < box_threshold:
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
