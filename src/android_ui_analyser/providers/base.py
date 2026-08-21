"""Provider Strategy interfaces + shared value objects (PRD §7.1).

Five provider *kinds*, each an abstract base class (a Strategy):

    OcrProvider.recognize(image)            -> list[TextBox]
    DetectionProvider.detect(image)         -> list[DetBox]
    GroundingProvider.locate(image, instr)  -> Point | DetBox | None
    GroundingProvider.parse(image)          -> list[DetBox] | None   (optional)
    GroundingProvider.ask(image, question, elements) -> ScreenAnalysisResult | None
    PlannerProvider.decide(objective, els)  -> PlannerDecision | None
    PolicyProvider.select(context)          -> candidate id | None

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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from PIL import Image as PILImage

    from ..policy import PolicyContext


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


@dataclass(frozen=True)
class ScreenAnalysisResult:
    """Provider-neutral result for a screenshot + UI-graph question.

    Providers keep their HTTP request/response details private and return this common
    shape to the engine. ``analysis`` is the structured screen answer; the remaining
    fields are diagnostic metadata exposed by ``aua ask``.
    """

    analysis: dict[str, Any]
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    input_image: dict[str, Any] = field(default_factory=dict)


# The action vocabulary a planner may emit. ``done``/``give-up`` are terminal; the rest
# name a state-changing action the engine already knows how to perform.
PLANNER_ACTIONS = frozenset({"tap", "input", "key", "swipe", "scroll-to", "done", "give-up"})


@dataclass(frozen=True)
class PlannerDecision:
    """One decision from a planner: the next action toward the objective (or a verdict).

    ``target_id`` is an id **from the element list handed to the planner** — the engine
    validates it against that set, so the model can never invent an off-screen target.
    """

    action: str  # one of PLANNER_ACTIONS
    target_id: int | None = None  # element id for tap/input (must be in the provided set)
    text: str | None = None  # value for `input`
    arg: str | None = None  # key name / swipe direction / scroll-to query
    reason: str | None = None  # short rationale (logs / enriched handoff)


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

    def ask(
        self,
        image: ScreenImage,
        question: str,
        elements: list[dict[str, Any]],
    ) -> ScreenAnalysisResult | None:
        """Answer a free-form screen question using pixels plus the element graph.

        This is optional so coordinate-only grounding providers remain valid. Implementors
        return the common result type so the engine never depends on provider response shapes.
        """
        return None


class PlannerProvider(Provider):
    kind: ClassVar[str] = "planner"

    @abstractmethod
    def decide(
        self,
        objective: str,
        elements: list[dict[str, Any]],
        image: ScreenImage | None = None,
    ) -> PlannerDecision | None:
        """Choose the next action toward *objective* given the on-screen *elements*.

        *elements* is a token-light list of ``{id, label, clickable, ...}`` from the
        current screen; *image* is attached only when the screen is weakly labelled.
        Returns ``None`` when the provider cannot decide (the chain then advances / the
        caller hands off) — like the other strategies, this must never raise.
        """
        raise NotImplementedError


class PolicyProvider(Provider):
    """Select one opaque ID from deterministic, already-guarded exact calls."""

    kind: ClassVar[str] = "policy"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        super().__init__(settings)
        self._model_monitor: Any = None

    def set_model_monitor(self, monitor: Any) -> None:
        """Attach the daemon-local observer used by the dashboard model workspace."""

        self._model_monitor = monitor

    def emit_model_event(self, event: Mapping[str, Any]) -> None:
        monitor = self._model_monitor
        if callable(monitor):
            monitor({"provider": self.name, **dict(event)})

    def load_model(self) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not expose model loading")

    def unload_model(self) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not expose model unloading")

    def interact(
        self, messages: list[dict[str, str]], *, max_tokens: int | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not expose direct interaction")

    def supports_candidate_count(self, count: int) -> bool:
        """Whether this provider accepts the guarded candidate cardinality."""

        return count > 0

    def supports_mode(self, mode: str) -> bool:
        """Whether provider provenance permits shadow/advisory rollout."""

        return mode in {"shadow", "advisory"}

    @abstractmethod
    def select(self, context: PolicyContext) -> int | None:
        """Return one supplied candidate ID, or ``None`` on any failure.

        The provider never executes or rewrites a call.  The policy core validates
        the returned ID against its trusted candidate map before exposing it.
        """
        raise NotImplementedError


@dataclass
class ChainSpec:
    """A resolved, ordered list of provider instances for one kind."""

    kind: str
    providers: list[Provider] = field(default_factory=list)

    def names(self) -> list[str]:
        return [p.name for p in self.providers]
