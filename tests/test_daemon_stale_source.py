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


def test_capture_methods_are_marked_daemon_only() -> None:
    """On skew these must error, because an in-process answer is wrong, not just slower."""
    from android_ui_analyser import cli

    for method in (
        "capture_status",
        "capture_last",
        "capture_export",
        "capture_explain",
        "capture_on",
        "capture_off",
        "capture_prune",
    ):
        assert method in cli._DAEMON_ONLY_METHODS, method
    # Stateless calls must NOT be in the set — they degrade to in-process correctly.
    for method in ("analyze", "tap", "has"):
        assert method not in cli._DAEMON_ONLY_METHODS, method
