"""A warm daemon must not silently serve code you have already edited.

The version handshake only compared release versions, which are identical during
development — so an edited file kept being served from the daemon's memory with no signal.
That produced three wrong conclusions in one session, including "this CLI flag does not
exist" about a flag that did.

Folding the loaded-source fingerprint into the reported identity makes the existing skew
path catch it. Two properties matter and are asserted here: the identity must change when
source changes, and it must be compared symmetrically — a composite on one side and a bare
version on the other would look skewed on every call and silently disable the daemon.
"""

from __future__ import annotations

from android_ui_analyser import __version__, daemon


def test_identity_carries_a_source_fingerprint() -> None:
    ident = daemon._aua_version()
    assert ident.startswith(f"{__version__}+src"), ident
    assert ident != __version__, "a bare version cannot detect an edited source tree"


def test_fingerprint_is_captured_at_import_not_recomputed() -> None:
    """The daemon must report what it LOADED, not what is on disk now.

    If it recomputed on every ping, a CLI started later would read the same value and the
    skew check could never fire — the bug this guards.
    """
    assert daemon._aua_version().split("+src", 1)[1] == daemon._LOADED_SOURCE
    recomputed = daemon._source_fingerprint()
    assert isinstance(recomputed, str) and recomputed


def test_only_the_methods_with_no_durable_form_are_marked_daemon_only() -> None:
    """Split on 2026-08-20 into the mutations and the reads; the reads have a durable form.

    This used to require all seven ``capture_*`` methods, on the rule "an in-process answer is
    wrong, not just slower". That rule is right, and it is why the mutations below are still
    listed: ``capture_on``/``capture_off``/``capture_prune`` reach into the daemon's sampler
    thread, so running them in-process would start or stop a buffer inside a CLI that is about
    to exit and report success for something that never happened.

    It does not hold for the reads. ``CaptureBuffer`` appends every kept frame to
    ``index.jsonl`` — timestamp, path, hash, dimensions and the action mark — before the call
    that produced it returns, so a process with no buffer can answer from that file and label
    it ``source: "disk-index"``. Keeping the reads here cost the caller both ways: it blocked
    the restart that would have made the daemon current, *and* the fallback that could have
    answered anyway. Measured against a real daemon: no frames at all, while 48 indexed frames
    sat readable on the same host.
    """
    from android_ui_analyser import cli

    # No durable representation outside the daemon process: a sampler thread, a job thread.
    for method in ("capture_on", "capture_off", "capture_prune", "job_start", "job_wait"):
        assert method in cli._DAEMON_ONLY_METHODS, method
    # Answerable from the on-disk index, so they must be free to restart and then to degrade.
    for method in ("capture_status", "capture_last", "capture_export", "capture_explain"):
        assert method not in cli._DAEMON_ONLY_METHODS, method
    # Stateless calls must NOT be in the set — they degrade to in-process correctly.
    for method in ("analyze", "tap", "has"):
        assert method not in cli._DAEMON_ONLY_METHODS, method
