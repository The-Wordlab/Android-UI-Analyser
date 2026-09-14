"""A verdict on its own is a claim; what makes it reviewable is what came back beside it.

Callers used to re-walk the artifact directory and guess which files mattered, which is how a full
logcat reaches a published bundle. These pin the grouping, the flagging, and the one thing the
payload must never imply: that AUA has vetted the contents for you.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.evidence import classify, collect, handback, needs_review


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "screenshots").mkdir(parents=True)
    (root / "screenshots" / "01-hub.png").write_bytes(b"\x89PNG" + b"0" * 60)
    (root / "screenshots" / "02-badge.png").write_bytes(b"\x89PNG" + b"0" * 40)
    (root / "run.mp4").write_bytes(b"\x00" * 100)
    (root / "report.md").write_text("# AUA session report\n")
    (root / "result.json").write_text("{}\n")
    (root / "calls.jsonl").write_text("{}\n")
    (root / "logcat.txt").write_text("device log\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    return root


def test_everything_a_run_produced_is_grouped_by_what_it_is(tmp_path) -> None:
    bundle = collect(_bundle(tmp_path))

    assert bundle["counts"] == {"image": 2, "video": 1, "text": 2, "data": 2, "other": 0}
    assert {entry["name"] for entry in bundle["images"]} == {
        "screenshots/01-hub.png",
        "screenshots/02-badge.png",
    }
    assert bundle["videos"][0]["name"] == "run.mp4"
    assert bundle["total_bytes"] > 0
    assert not bundle["truncated"]


def test_build_noise_is_not_offered_as_evidence(tmp_path) -> None:
    names = {entry["name"] for entry in collect(_bundle(tmp_path))["data"]}
    assert not any(name.startswith("__pycache__") for name in names)


def test_device_and_network_records_are_flagged_rather_than_dropped(tmp_path) -> None:
    bundle = collect(_bundle(tmp_path))

    # Flagged - a disputed verdict is exactly when you want the raw calls.
    assert bundle["review_before_publishing"] == ["calls.jsonl", "logcat.txt"]
    assert {entry["name"] for entry in bundle["text"]} == {"report.md", "logcat.txt"}
    assert any(entry["name"] == "calls.jsonl" for entry in bundle["data"])


def test_a_recording_written_outside_the_bundle_is_still_listed(tmp_path) -> None:
    root = _bundle(tmp_path)
    elsewhere = tmp_path / "somewhere" / "screen.mp4"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"\x00" * 10)

    bundle = collect(root, extra=[elsewhere, tmp_path / "missing.mp4"])

    assert {entry["name"] for entry in bundle["videos"]} == {"run.mp4", "screen.mp4"}


def test_an_extra_file_already_inside_the_bundle_is_not_counted_twice(tmp_path) -> None:
    root = _bundle(tmp_path)
    bundle = collect(root, extra=[root / "run.mp4"])
    assert len(bundle["videos"]) == 1


def test_the_handback_says_plainly_that_nobody_has_read_the_flagged_files(tmp_path) -> None:
    root = _bundle(tmp_path)
    payload = handback(
        {"verdict": "passed", "goal": "badge shows once"}, root=root, report=root / "report.md"
    )

    assert payload["verdict"] == "passed"
    assert payload["evidence"]["counts"]["image"] == 2
    assert payload["evidence"]["report"].endswith("report.md")
    assert "AUA has not read them for you" in payload["publishing"]


def test_a_clean_bundle_still_warns_about_what_is_on_screen(tmp_path) -> None:
    root = tmp_path / "clean"
    root.mkdir()
    (root / "shot.png").write_bytes(b"\x89PNG")

    payload = handback({"verdict": "passed"}, root=root)

    assert payload["evidence"]["review_before_publishing"] == []
    assert "whatever was on screen" in payload["publishing"]


def test_a_missing_directory_is_an_empty_bundle_not_a_crash(tmp_path) -> None:
    bundle = collect(tmp_path / "never-ran")
    assert bundle["counts"] == {"image": 0, "video": 0, "text": 0, "data": 0, "other": 0}
    assert bundle["review_before_publishing"] == []


def test_the_file_kinds_are_decided_by_extension_alone(tmp_path) -> None:
    assert classify(Path("a.PNG")) == "image"
    assert classify(Path("a.mp4")) == "video"
    assert classify(Path("a.yaml")) == "data"
    assert classify(Path("a.bin")) == "other"
    assert needs_review(Path("out/logcat-run.txt"))
    assert not needs_review(Path("out/report.md"))
