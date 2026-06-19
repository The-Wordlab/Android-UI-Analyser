"""Provider Strategy interfaces + shared value objects (PRD §7.1).

Three provider *kinds*, each an abstract base class (a Strategy):

    OcrProvider.recognize(image)            -> list[TextBox]
    DetectionProvider.detect(image)         -> list[DetBox]
    GroundingProvider.locate(image, instr)  -> Point | DetBox | None
    GroundingProvider.parse(image)          -> list[DetBox] | None   (optional)

The engine depends ONLY on these interfaces and on the factory (registry.py). It never
imports a concrete provider. Adding a model = implement a strategy + register it +
add a ``models.<name>`` config block; ZERO changes to engine.py / cli.py.

Value objects (``ScreenImage``, ``TextBox``, ``DetBox``, ``Point``) are deliberately
plain dataclasses with no pydantic/network deps so providers stay light. Heavy deps
(torch, pyobjc, onnxruntime, …) must be lazy-imported inside ``is_available()`` / on
first use so a missing optional dependency never breaks the core CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from PIL import Image as PILImage


Bounds = tuple[int, int, int, int]  # (x1, y1, x2, y2)


# --------------------------------------------------------------------------- image


class ScreenImage:
    """A captured screen, decoded lazily.

    Carries the raw PNG bytes and exposes whatever representation a provider needs
    (PIL image, RGB numpy array, on-disk path) without forcing a re-encode. PIL and
    numpy are base dependencies but imported lazily to keep import time low.
    """

    __slots__ = ("_png", "_pil", "_np", "_path", "_width", "_height")

    def __init__(
        self,
        png_bytes: bytes,
        *,
        width: int | None = None,
        height: int | None = None,
        path: str | None = None,
    ) -> None:
        self._png = png_bytes
        self._pil: PILImage.Image | None = None
        self._np: np.ndarray | None = None
        self._path = path
        self._width = width
        self._height = height

    @property
    def png_bytes(self) -> bytes:
        return self._png

    @property
    def path(self) -> str | None:
        return self._path

    def pil(self) -> PILImage.Image:
        if self._pil is None:
            import io

            from PIL import Image

            self._pil = Image.open(io.BytesIO(self._png)).convert("RGB")
            self._width, self._height = self._pil.size
        return self._pil

    def numpy(self) -> np.ndarray:
        """RGB uint8 array, shape (H, W, 3)."""
        if self._np is None:
            import numpy as np

            self._np = np.asarray(self.pil())
        return self._np

    def _ensure_size(self) -> None:
        if self._width is None or self._height is None:
            self.pil()  # populates size as a side effect

    @property
    def width(self) -> int:
        self._ensure_size()
        assert self._width is not None
        return self._width

    @property
    def height(self) -> int:
        self._ensure_size()
        assert self._height is not None
        return self._height

    def save(self, path: str) -> str:
        with open(path, "wb") as fh:
            fh.write(self._png)
        self._path = path
        return path

    @classmethod
    def from_pil(cls, img: PILImage.Image) -> ScreenImage:
        import io

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return cls(buf.getvalue(), width=img.size[0], height=img.size[1])


# ------------------------------------------------------------------- value objects


@dataclass(frozen=True)
class TextBox:
    """A recognised line/word of text and its pixel box."""

    text: str
    bounds: Bounds
    confidence: float | None = None


@dataclass(frozen=True)
class DetBox:
    """A detected (or VLM-parsed) box, optionally labelled/interactable."""

    bounds: Bounds
    label: str | None = None
    interactable: bool = True
    confidence: float | None = None


@dataclass(frozen=True)
class Point:
    """A grounded point (e.g. a VLM click target)."""

    x: int
    y: int
    confidence: float | None = None


class Availability(NamedTuple):
    """Result of ``Provider.is_available()`` — unpacks as ``(ok, reason)``."""

    ok: bool
    reason: str


# --------------------------------------------------------------------------- bases


class Provider(ABC):
    """Common base for every strategy.

    Subclasses set ``kind``/``name`` (the registry decorator also sets these) and read
    their settings from the ``models.<name>`` config block, passed in as ``settings``.
    """

    kind: ClassVar[str] = "provider"
    name: ClassVar[str] = "provider"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings: dict[str, Any] = dict(settings or {})

    @abstractmethod
    def is_available(self) -> Availability:
        """Cheap check: deps importable, platform OK, key present, endpoint set.

        Must NOT do network round-trips by default and must NOT raise.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.kind}:{self.name}>"


class OcrProvider(Provider):
    kind: ClassVar[str] = "ocr"

    @abstractmethod
    def recognize(self, image: ScreenImage) -> list[TextBox]:
        """Return recognised text boxes (may be empty)."""
        raise NotImplementedError


class DetectionProvider(Provider):
    kind: ClassVar[str] = "detection"

    @abstractmethod
    def detect(self, image: ScreenImage) -> list[DetBox]:
        """Return detected interactable boxes (may be empty)."""
        raise NotImplementedError


class GroundingProvider(Provider):
    kind: ClassVar[str] = "grounding"

    @abstractmethod
    def locate(self, image: ScreenImage, instruction: str) -> Point | DetBox | None:
        """Map a natural-language instruction to a point/box, or None if not found."""
        raise NotImplementedError

    def parse(self, image: ScreenImage) -> list[DetBox] | None:
        """Optional: full-screen parse for VLMs that can enumerate elements.

        Default ``None`` means "this provider does not support parse"; the engine then
        skips it for the vision-parse path.
        """
        return None


@dataclass
class ChainSpec:
    """A resolved, ordered list of provider instances for one kind."""

    kind: str
    providers: list[Provider] = field(default_factory=list)

    def names(self) -> list[str]:
        return [p.name for p in self.providers]
