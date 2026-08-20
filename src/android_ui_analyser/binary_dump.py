"""Host-side binary screen dumps (phase 3 stand-in for on-device FlatBuffers).

True FlatBuffers need a device-side producer (phase 2 AccessibilityService / u2 fork).
Until then we ship:

* ``schemas/hierarchy.fbs`` — the target schema (documentation + future codegen)
* this module — zlib-framed msgpack-or-JSON payloads for ``--format msgpack``

Wire format (little-endian)::

    magic[4] = b"AUA1"
    flags[1]  = 0 JSON  | 1 msgpack (if ``msgpack`` importable)
    length[4] = uint32 payload size
    payload   = zlib-compressed JSON object (or msgpack map) of AnalyzeResult.as_dict(compact)
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

MAGIC = b"AUA1"

# Declared before the import so the fallback needs no `type: ignore`. Suppressing the
# module-vs-None assignment only type-checks on hosts where msgpack is *absent*: with it
# installed the ignore is required, without it mypy calls the same ignore unused, so the
# check could not pass in both environments at once.
msgpack: Any = None
_HAS_MSGPACK = False
try:
    import msgpack as _msgpack
except ImportError:  # pragma: no cover - optional
    pass
else:
    msgpack = _msgpack
    _HAS_MSGPACK = True


def pack_analyze_dict(data: dict[str, Any]) -> bytes:
    """Encode a compact analyze dict to the AUA1 binary frame."""
    if _HAS_MSGPACK:
        raw = msgpack.packb(data, use_bin_type=True)
        flags = 1
    else:
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
        flags = 0
    payload = zlib.compress(raw, level=6)
    return MAGIC + struct.pack("<BI", flags, len(payload)) + payload


def pack_analyze(result: Any) -> bytes:
    from .schema import OutputFormat

    data = result.as_dict(OutputFormat.compact)
    return pack_analyze_dict(data)


def pack_analyze_b64(result: Any) -> str:
    """Base64 text form so CLI stdout stays a single line."""
    return base64.b64encode(pack_analyze(result)).decode("ascii")


def unpack_frame(blob: bytes) -> dict[str, Any]:
    if len(blob) < 9 or blob[:4] != MAGIC:
        raise ValueError("not an AUA1 frame")
    flags, length = struct.unpack_from("<BI", blob, 4)
    payload = blob[9 : 9 + length]
    raw = zlib.decompress(payload)
    if flags == 1:
        if not _HAS_MSGPACK:
            raise ValueError("frame is msgpack but msgpack is not installed")
        out = msgpack.unpackb(raw, raw=False)
    else:
        out = json.loads(raw.decode())
    if not isinstance(out, dict):
        raise ValueError("AUA1 payload is not an object")
    return out


def unpack_b64(text: str) -> dict[str, Any]:
    return unpack_frame(base64.b64decode(text.encode("ascii")))
