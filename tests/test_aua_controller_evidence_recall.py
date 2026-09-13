"""Fictional local archives only; no device, provider, or network access."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.evidence_recall import MAX_RECORD_BYTES, read_recorded_evidence
from experiments.aua_controller.session_state import observation_frame


def save(root, ref="E0001", value=None, raw=None):
    directory = root / "evidence"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (ref + ".json")
    path.write_bytes(raw if raw is not None else json.dumps(value or {"ok": True}).encode() + b"\n")
    return path


def test_current_record_retains_exact_source_digest_but_is_never_a_current_frame(tmp_path):
    frame = {"screen": {"width": 400, "height": 800},
             "elements": [{"id": "el:fictional", "text": "Earlier screen"}],
             "meta": {"fingerprint": "formerly-fresh"}}
    source = save(tmp_path, value={"ok": True, "observation": frame})
    before = source.read_bytes()
    recalled = read_recorded_evidence("E0001", {"": tmp_path})
    assert recalled == {"ok": True, "historical_only": True, "source_evidence_ref": "E0001",
                        "source_sha256": hashlib.sha256(before).hexdigest(),
                        "historical_record": {"ok": True, "observation": frame}}
    assert observation_frame(recalled) is None
    assert source.read_bytes() == before


def test_qualified_prior_row_uses_exact_host_mapping(tmp_path):
    current, previous = tmp_path / "current", tmp_path / "previous"
    save(current, value={"text": "Current archive"})
    save(previous, value={"text": "Earlier archive"})
    roots = {"": current, "fictional-row-1": previous}
    result = read_recorded_evidence("fictional-row-1/E0001", roots)
    assert result["historical_record"] == {"text": "Earlier archive"}
    assert result["source_evidence_ref"] == "fictional-row-1/E0001"
    assert read_recorded_evidence("E0001", roots)["historical_record"] == {"text": "Current archive"}
    assert read_recorded_evidence("unmapped/E0001", roots)["error"]["code"] == "unknown_evidence_namespace"


@pytest.mark.parametrize("reference", [
    "", "E001", "E00001", "e0001", "E0001.json", " E0001", "E0001\n", "E٠٠٠١",
    "../E0001", "./E0001", "/E0001", "row/../E0001", "row//E0001", "row\\E0001",
    "file:///private/E0001", "C:/E0001", "row%2F..%2FE0001", "row/E0001/extra", None, {},
])
def test_malformed_references_never_echo_input(reference, tmp_path):
    assert read_recorded_evidence(reference, {"": tmp_path}) == {
        "ok": False, "historical_only": True, "error": {"code": "invalid_evidence_reference"}}


@pytest.mark.parametrize("link_directory", [False, True])
def test_symlink_escape_is_rejected_without_reading_private_record(tmp_path, link_directory):
    root, outside = tmp_path / "approved", tmp_path / "outside"
    root.mkdir()
    target = save(outside, value={"text": "Must not be returned"})
    if link_directory:
        (root / "evidence").symlink_to(target.parent, target_is_directory=True)
    else:
        (root / "evidence").mkdir()
        (root / "evidence/E0001.json").symlink_to(target)
    result = read_recorded_evidence("E0001", {"": root})
    assert result == {"ok": False, "historical_only": True, "error": {"code": "evidence_unavailable"}}


def test_in_root_symlink_and_missing_record(tmp_path):
    source = save(tmp_path, ref="E0002", value={"text": "Stored locally"})
    (source.parent / "E0001.json").symlink_to(source)
    assert read_recorded_evidence("E0001", {"": tmp_path})["ok"] is True
    assert read_recorded_evidence("E0003", {"": tmp_path})["error"]["code"] == "evidence_unavailable"


def test_size_limit_is_bytes_and_accepts_exact_boundary(tmp_path):
    raw = b'{"text":"' + b"x" * (MAX_RECORD_BYTES - 11) + b'"}'
    assert len(raw) == MAX_RECORD_BYTES
    source = save(tmp_path, raw=raw)
    assert read_recorded_evidence("E0001", {"": tmp_path})["ok"] is True
    source.write_bytes(raw + b" ")
    assert read_recorded_evidence("E0001", {"": tmp_path})["error"]["code"] == "evidence_too_large"


@pytest.mark.parametrize("raw", [b"[]", b"null", b'"string"', b"{", b"\xff", b'{"value":NaN}'])
def test_non_object_or_invalid_json_has_fixed_error(tmp_path, raw):
    save(tmp_path, raw=raw)
    assert read_recorded_evidence("E0001", {"": tmp_path}) == {
        "ok": False, "historical_only": True, "error": {"code": "invalid_evidence_record"}}


def test_private_metadata_and_paths_projected_without_altering_source(tmp_path):
    source = save(tmp_path, value={"session_id": "fictional-private-session",
        "image_path": "/Users/fictional/private/screen.png",
        "note": "Read /Users/fictional/private/proof.json for fictional-private-session",
        "elements": [{"id": "el:retained", "resource_id": "dev.fictional:id/button", "text": "Continue"}]})
    before = source.read_bytes()
    result = read_recorded_evidence("E0001", {"": tmp_path})
    encoded = json.dumps(result)
    assert "fictional-private-session" not in encoded and "/Users/fictional" not in encoded
    assert result["historical_record"]["elements"][0]["id"] == "el:retained"
    assert result["historical_record"]["elements"][0]["resource_id"] == "dev.fictional:id/button"
    assert source.read_bytes() == before
