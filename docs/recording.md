# Native recording evidence

`aua record start` begins native Android encoding in an owned target directory. The
target-side supervisor rotates `screenrecord` processes at their 180-second limit, for
up to 30 minutes by default. The supervisor survives the initiating CLI or daemon process.
The write-ahead `screen_recording` undo covers the supervisor, encoders, logs and segments.
Normal stop or teardown stops rotation, signals only encoders with the exact owned output
path and waits for them to exit. A complete export permits removal of target artifacts;
partial salvage and unverified cleanup keep the originals and undo.

`aua record stop journey.mp4` saves a playable MP4 at `journey.mp4`. The response's
`detail` continues to name that MP4; `recording` includes the coverage metadata. AUA also
retains the original native files in `journey.mp4.segments/` and a sidecar report at
`journey.mp4.recording.json`. One segment is copied byte for byte. Multiple segments use
optional host `ffmpeg`, discovered on PATH, to concatenate in index order with bitstream
copy. AUA does not install it. Export uses local files, no shell, and a bounded timeout.

The playable export preserves native timing within each segment, including encoded pauses.
It joins captured media and **omits inter-segment wall-clock gaps**. Its duration is captured
media time, not proof that the requested wall-clock interval was covered. No frames are
stretched, sampled, or synthesized to fill missing footage. Original segment files remain
unchanged so the sidecar can be assessed against the requested recording window.

Missing ffmpeg returns `recording_export_unsupported`; export or publication failure returns
`recording_export_failed`. Remote originals and lifecycle events, the undo, and the original
stop-request time remain available. Retry `record stop` after fixing the export dependency.
Existing output files, symlinks, segment directories and sidecars are never overwritten.
If publication or remote cleanup failed after writing local evidence, keep that evidence and
retry at a new destination. A failed directory allocation does not authorize touching an
existing target directory: teardown requires its matching recording/boot identity, or a
positive check that the directory is absent. Unverified ownership retains the pending undo.

Inspect `segments`, `gaps`, `finish`, and `duration_check`. Lifecycle timestamps use target
boot uptime, so changing the device wall clock cannot shorten a measured pause. The native
movie duration is compared with the encoder process window and the requested recording
window, with a two-second tolerance. A duration shortfall or missing supervisor completion
returns `ok=false` after evidence collection. After encoder failure or incomplete supervisor
lifecycle, stop exports all finalized segments in order into the requested playable MP4.
Every lifecycle segment has a `status` (`finalized`, `missing`, or `corrupt`) and `exported`
flag. Corrupt/partial files are copied unchanged into the segment directory. Missing footage
is explicitly unexported; absence requires a successful inspection with a known missing-file
result, and transport or permission errors remain errors rather than being called missing.

Partial salvage returns failed coverage with `cleanup_pending=true` and `retained_remote`.
It keeps remote originals, diagnostics, local metadata and the undo for inspection or retry
at a new destination; deliberate teardown can later remove the remote copies while the
published originals remain. With no finalized segments, stop publishes the available original
bytes and failed sidecar, then raises `recording_no_playable_segments`. It never fabricates a
video. A failed pull or a corrupt download after an otherwise clean lifecycle preserves remote
files for retry. Retries use the original stop-request time.

A verified new boot or recycled serial can start recording again without deleting earlier
footage. The runtime checks full process identities and the old root's recorded boot identity
(or proves that root absent), then quarantines stale local metadata. The engine retains the
old undo under a `screen_recording:prior:` key, with its original target/boot and remote path.
Those older files remain on the target if storage survived reboot; they are not automatically
deleted or described as restored. Unknown identity, unreadable inspection, or a process using
the old root fails closed. Prior-boot undos still fail the normal boot guard on replay.

Process discovery requests wide Toybox `ps` output and resolves shell/encoder candidates
through one batched `/proc/<pid>/cmdline` read. An incomplete or ambiguous response prevents
cleanup, rather than authorizing deletion beneath a possible live encoder. Empty command
lines require matching zombie/dead process stat evidence before they can be ignored. The
background launch is a compound command so the transport can append its exit-status suffix
without invalidating shell syntax; nohup and target-side redirections remain in place.

The 30-minute default bounds time, not disk usage: native recording can consume several GB
on the target. Host storage retains the original segments plus the playable concatenation,
roughly twice the media size; failed exports, retained prior boots and repeated exports can
need more. There is no disk-space scheduler or automatic deletion of prior evidence.

**Gapless coverage is not guaranteed.** Rotation has gaps, and process timestamps do not
identify exact first/last encoded frame times. `continuous_coverage_verified` remains false,
even when the duration check passes. A native encoder can omit footage or exit early;
a finalized MP4 alone does not establish end-to-end coverage. A static or missing tail is
not assumed captured, and known gaps are not subtracted to obtain a passing duration check. If continuous evidence is
required, assess the manifest against that requirement and report uncovered intervals.
Rolling `capture export` animations are sampled evidence and cannot replace these recordings.

The core uses `device.recording` on the selected target runtime. The optional
`device.recording.timeline` capability supplies `recording_metadata()` for CLI and MCP through
the same engine methods. The optional `device.recording.recovery` capability supplies
`archive_stale_recording(path, instance_token)`; the selected adapter must prove a different
boot and inert prior path, quarantine metadata, and retain evidence. Adapters without recovery
keep a pending undo blocked rather than falling back to Android. Adapters without recording
fail explicitly; adapters with recording but no timeline cannot supply coverage evidence. Android implements the supervisor and native
process/file operations in `platforms/android_recording.py`, reached through `AndroidPlatform`.

Observation flags control action readback. They do not disable an independently running
recorder, rolling capture, screenshots used by actions, or journal persistence.

Native launch requires `setsid`. The launcher establishes hangup protection before forking, detaches the supervisor session, and waits up to 30 readiness checks (0.1 seconds apart). Missing detachment support or readiness fails explicitly and retains pending recovery. Readiness alone does not prove capture: the existing live-process, lifecycle-event and segment-file checks still apply.

Concatenated exports copy video and optional audio streams. Native non-media tracks are retained in the original segment files; the sidecar reports this stream policy. A single-segment direct copy retains the complete original file.
