package dev.aua.helper;

import android.accessibilityservice.AccessibilityService;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * {@code record.*} — capture what a user actually did, as AUA steps.
 *
 * <p>AUA records journeys by watching polled snapshots and inferring what must have happened
 * between them. The device does not have to infer: the framework says "this node was clicked"
 * at the moment it happens, with the node attached. That is a better source for the same
 * ``RouteStep`` list — no sampling gap to fall through, and no guessing which of several
 * changes was the one the user caused.
 *
 * <p>Steps come out shaped exactly like the host's ``RouteStep`` — ``kind`` plus a durable
 * selector, ``resource_id`` tail first and ``label`` second — so no translation is needed to
 * read one. Whether a recording is *complete* enough to replay is a separate question, and
 * the answer today is no; see below.
 *
 * <p>Typed values are deliberately never captured. An auto-recorded step must be safe to save
 * and share, and what someone typed into a field is exactly what must not be.
 *
 * <h3>Incomplete by construction, and it says so — measured</h3>
 *
 * <p>Scrolls and text changes arrive reliably. Taps do not: {@code TYPE_VIEW_CLICKED} is only
 * sent when a view actually calls {@code performClick}, and plenty do not. Tapping a row in
 * the emulator's app list produced 35 {@code TYPE_WINDOW_CONTENT_CHANGED} events and no click
 * event at all, while a tap on a search bar in the same app did report one.
 *
 * <p>That is why this was left unreachable for a long time: a recording is a faithful log of
 * what the framework announced, not of what the user did, so replaying one could silently skip
 * a step. Silence is ambiguous — it means either "nothing happened" or "something happened
 * that I cannot see" — and a draft flow missing a step looks exactly like a finished one.
 *
 * <p>What makes it usable is that the ambiguity is removable without touch exploration. A
 * missed tap does not announce itself, but it leaves a shadow: the window changes state with
 * nothing announced before it. Recording that shadow as an explicit {@code gap} step turns
 * "I might have missed something" into "I went blind here, between these two steps", which
 * the host refuses to save as a runnable flow. Touch exploration
 * ({@code canRequestTouchExplorationMode}) would close the gap properly but changes how the
 * device behaves for the person using it, so it stays off; being honest is cheaper than being
 * complete, and for an authoring aid it is enough.
 */
final class RecordFeature implements Feature {

    /** Bounded so a recording left running cannot grow without limit. */
    private static final int MAX_STEPS = 2000;

    private final List<JSONObject> steps = Collections.synchronizedList(new ArrayList<>());
    private volatile boolean recording = false;
    private volatile String lastScrollSignature = "";
    /**
     * Has anything been *announced* since the screen last changed?
     *
     * <p>This is what turns the blind spot below into a reportable one. A window state change
     * with no announced action before it means the user did something the framework never told
     * us about — the missed tap, in other words, leaves a shadow even though the tap itself
     * does not. Recording that shadow is the difference between a draft that silently skips a
     * step and one that says exactly where it stopped being able to see.
     */
    private volatile boolean announcedSinceWindowChange = false;

    /**
     * When the screen last changed state, so one navigation is not counted as several.
     *
     * <p>A single tap does not produce a single window change. Opening one stock settings page
     * produced a burst of them, and crediting only the first to the tap made every later one
     * in the burst look like an unexplained navigation — the recorder reported two holes in a
     * three-action journey that had none. A flag that fires on a clean recording is worse than
     * no flag, because it teaches the reader to skip past the one that matters.
     */
    private volatile long lastWindowChangeMs = 0L;

    /** How long a navigation may keep settling before a further change counts as a new one. */
    private static final long WINDOW_SETTLE_MS = 800L;

    /**
     * Grace after arming, during which a screen change is the recorder arriving, not a user.
     *
     * <p>Arming necessarily lands mid-frame, so the first window change is usually noise. That
     * used to be handled on the host by dropping any gap before the first recorded step — and
     * that rule was wrong in the case that matters. Start recording on an app that is already
     * open, tap a row Android does not announce, and the very first thing to happen is a real
     * missed action arriving with no step before it. The host dropped it, so the recording
     * claimed to be complete while missing its opening step. Time since arming distinguishes
     * the two; position in the list does not.
     */
    private static final long ARMING_GRACE_MS = 1500L;

    /** When record.start ran, so arming noise can be told apart from a user's first action. */
    private volatile long recordingStartedMs = 0L;

    /**
     * Rolling snapshots of what was pressable, so a bare coordinate can be given a name.
     *
     * <p>The host can read the kernel touch stream and knows exactly where a finger landed,
     * including for the taps Android never announces. A coordinate on its own only makes a
     * brittle `tap-point` step, though — what turns it into a real selector is knowing what
     * was under it *at that moment*, and by the time anyone asks, the screen has moved on.
     *
     * <p>So the pressable nodes are captured as the screen changes and kept with their
     * timestamps. The host then looks up the newest snapshot taken before the touch. Only
     * clickable nodes are kept, and only a bounded number of them, because this runs on the
     * accessibility callback thread while a person is actually using the device.
     */
    private final List<JSONObject> snapshots = Collections.synchronizedList(new ArrayList<>());

    private volatile long lastSnapshotMs = 0L;
    /**
     * Why snapshots did or did not happen, reported in the drain.
     *
     * <p>Kept because it is what found the bug this mechanism exists to fix: the counters
     * said 47 calls and 2 stored, which is how the event-driven trigger was shown to be
     * starving. A recorder that quietly captures nothing is the failure mode here, and
     * these are the only numbers that distinguish it from a screen nobody touched.
     */
    private volatile int diagCalls = 0;
    private volatile int diagThrottled = 0;
    private volatile int diagNullRoot = 0;
    private volatile int diagEmpty = 0;
    private volatile int diagOk = 0;

    /** Cheap enough to run between a person's taps; short enough to precede one. */
    private static final long SNAPSHOT_THROTTLE_MS = 350L;

    /**
     * Snapshots are driven by a clock, not by accessibility events, and that is the whole point.
     *
     * <p>Events looked like the natural trigger: the device telling us the screen changed, with
     * no thread to own. Measured on a real Compose app, a full navigation from home to settings
     * produced exactly ONE event — the same silence that stops it announcing taps. So every
     * press after the first resolved against a snapshot of the screen the journey started on,
     * and came back as bare coordinates.
     *
     * <p>An app that cannot be relied on to say when it changed cannot be relied on to drive
     * this either. A ticker costs one sleeping thread for the length of a recording somebody
     * asked for, and unchanged screens are deduplicated below, so a still screen costs a tree
     * walk and nothing else.
     */
    private volatile Thread ticker = null;

    /** Signature of the last stored snapshot, so a still screen is not stored sixty times. */
    private volatile String lastSnapshotShape = "";

    private static final int MAX_SNAPSHOTS = 60;

    private static final int MAX_SNAPSHOT_NODES = 80;

    @Override
    public String namespace() {
        return "record";
    }

    @Override
    public JSONObject handle(String method, JSONObject params) throws Exception {
        switch (method) {
            case "record.start":
                steps.clear();
                snapshots.clear();
                lastScrollSignature = "";
                announcedSinceWindowChange = false;
                lastWindowChangeMs = 0L;
                lastSnapshotMs = 0L;
                lastSnapshotShape = "";
                diagCalls = diagThrottled = diagNullRoot = diagEmpty = diagOk = 0;
                recordingStartedMs = System.currentTimeMillis();
                recording = true;
                // Snapshot the screen as it is right now. Snapshots are otherwise driven by
                // accessibility events, and a screen nobody has touched yet produces none —
                // so the very first press, on the screen recording started from, had nothing
                // to be resolved against and came back as bare coordinates. That is the one
                // press most likely to exist in any recording.
                snapshotIfDue(recordingStartedMs);
                startTicker();
                return new JSONObject().put("recording", true);
            case "record.stop": {
                recording = false;
                stopTicker();
                return drain();
            }
            case "record.peek":
                return drain();
            default:
                throw new IllegalArgumentException("unknown method: " + method);
        }
    }

    private JSONObject drain() throws Exception {
        JSONArray out = new JSONArray();
        synchronized (steps) {
            for (JSONObject s : steps) {
                out.put(s);
            }
        }
        JSONArray shots = new JSONArray();
        synchronized (snapshots) {
            for (JSONObject shot : snapshots) {
                shots.put(shot);
            }
        }
        return new JSONObject()
                .put("recording", recording)
                .put("count", out.length())
                .put("steps", out)
                .put("started_ms", recordingStartedMs)
                .put("diag", new JSONObject().put("calls", diagCalls)
                        .put("throttled", diagThrottled).put("null_root", diagNullRoot)
                        .put("empty", diagEmpty).put("ok", diagOk))
                .put("snapshots", shots);
    }

    /**
     * Record what is pressable right now, if enough time has passed since the last one.
     *
     * <p>Driven by whatever events happen to arrive rather than a timer, because the
     * accessibility callback is already the device telling us the screen changed — a timer
     * would only add a thread to learn the same thing later.
     */
    private void startTicker() {
        stopTicker();
        Thread t = new Thread(() -> {
            while (recording && !Thread.currentThread().isInterrupted()) {
                try {
                    Thread.sleep(SNAPSHOT_THROTTLE_MS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
                try {
                    snapshotIfDue(System.currentTimeMillis());
                } catch (Exception ignored) {
                    // a tree torn down mid-walk is not worth ending the recording over
                }
            }
        }, "aua-record-snapshots");
        t.setDaemon(true);
        ticker = t;
        t.start();
    }

    private void stopTicker() {
        Thread t = ticker;
        ticker = null;
        if (t != null) {
            t.interrupt();
        }
    }

    private void snapshotIfDue(long now) {
        diagCalls++;
        if (now - lastSnapshotMs < SNAPSHOT_THROTTLE_MS) {
            diagThrottled++;
            return;
        }
        // A bounded wait rather than none. record.start fires this immediately after the host
        // has handed the UiAutomation slot back, and the service is often a moment away from
        // having a usable tree — so the opening snapshot came back empty and the first press
        // of the journey, on the screen recording started from, had nothing to be resolved
        // against. It returns instantly whenever the service is already up, which is every
        // other call site.
        AccessibilityService service = HelperService.awaitService(1500L);
        AccessibilityNodeInfo root = service == null ? null : service.getRootInActiveWindow();
        if (root == null) {
            diagNullRoot++;
            // Deliberately NOT marking the attempt: a snapshot that could not be taken must
            // be retried on the next event, not throttled out for another 350ms.
            return;
        }
        lastSnapshotMs = now;
        JSONArray nodes = new JSONArray();
        try {
            collectPressable(root, nodes);
        } catch (Exception e) {
            return;   // a torn-down tree mid-walk is not worth a partial snapshot
        }
        if (nodes.length() == 0) {
            diagEmpty++;
            lastSnapshotMs = 0L;   // nothing pressable yet; look again on the next event
            return;
        }
        // A still screen produces an identical node set every tick. Storing each one would
        // push the screens that actually matter out of a bounded ring within seconds.
        String shape = nodes.toString();
        if (shape.equals(lastSnapshotShape)) {
            return;
        }
        lastSnapshotShape = shape;
        try {
            synchronized (snapshots) {
                if (snapshots.size() >= MAX_SNAPSHOTS) {
                    snapshots.remove(0);
                }
                snapshots.add(new JSONObject().put("ts", now).put("nodes", nodes));
                diagOk++;
            }
        } catch (Exception ignored) {
            // never let bookkeeping break the recording
        }
    }

    private static void collectPressable(AccessibilityNodeInfo node, JSONArray out)
            throws Exception {
        if (node == null || out.length() >= MAX_SNAPSHOT_NODES) {
            return;
        }
        if (node.isClickable() || node.isLongClickable()) {
            android.graphics.Rect b = new android.graphics.Rect();
            node.getBoundsInScreen(b);
            if (b.width() > 0 && b.height() > 0) {
                JSONObject entry = new JSONObject();
                String rid = node.getViewIdResourceName();
                if (rid != null && !rid.isEmpty()) {
                    int slash = rid.indexOf('/');
                    entry.put("resource_id", slash >= 0 ? rid.substring(slash + 1) : rid);
                }
                CharSequence desc = node.getContentDescription();
                if (desc != null && desc.length() > 0) {
                    entry.put("content_desc", desc.toString());
                }
                CharSequence text = node.getText();
                if (text != null && text.length() > 0) {
                    entry.put("label", text.toString());
                } else if (desc == null || desc.length() == 0) {
                    String rolled = rolledUpLabel(node);
                    if (rolled != null) {
                        entry.put("label", rolled);
                    }
                }
                entry.put("bounds", new JSONArray()
                        .put(b.left).put(b.top).put(b.right).put(b.bottom));
                out.put(entry);
            }
        }
        for (int i = 0; i < node.getChildCount(); i++) {
            collectPressable(node.getChild(i), out);
        }
    }

    /** Cap on a rolled-up label, matching the host's ``_MAX_LABEL``. */
    private static final int MAX_LABEL = 120;

    /** How far up to look for the clickable ancestor a label really belongs to. */
    private static final int MAX_ADDRESSABLE_DEPTH = 8;

    /**
     * The node the host's projection would have kept for this press.
     *
     * <p>``parse_hierarchy`` absorbs an unlabelled, non-actionable node into its clickable
     * ancestor, so that ancestor is what an agent sees and what a selector must name. A click
     * event's source is sometimes the inner view instead, and recording that produced a step
     * pointing at an element no ``analyze`` result contains.
     */
    private static AccessibilityNodeInfo addressable(AccessibilityNodeInfo node) {
        AccessibilityNodeInfo n = node;
        for (int depth = 0; n != null && depth < MAX_ADDRESSABLE_DEPTH; depth++) {
            if (n.isClickable() || n.isLongClickable()) {
                return n;
            }
            AccessibilityNodeInfo parent = n.getParent();
            if (parent == null) {
                break;
            }
            n = parent;
        }
        return node;
    }

    /**
     * Join a subtree's text and descriptions, exactly as the host's
     * ``_gather_descendant_text`` does: document order, case-insensitive de-dupe, capped.
     *
     * <p>Matching that rule character for character is the whole point. The recorded label is
     * replayed by equality against what ``analyze`` reports, so a roll-up that merely looks
     * reasonable — a different separator, no de-dupe, a different cap — produces a selector
     * that is never going to match the screen it was recorded from.
     */
    private static String rolledUpLabel(AccessibilityNodeInfo node) {
        StringBuilder out = new StringBuilder();
        java.util.Set<String> seen = new java.util.LinkedHashSet<>();
        collectLabels(node, seen);
        for (String part : seen) {
            if (out.length() > 0) {
                out.append(' ');
            }
            out.append(part);
        }
        String label = out.toString().trim();
        if (label.isEmpty()) {
            return null;
        }
        return label.length() > MAX_LABEL ? label.substring(0, MAX_LABEL) : label;
    }

    private static void collectLabels(AccessibilityNodeInfo node, java.util.Set<String> seen) {
        if (node == null || seen.size() > 64) {
            return;
        }
        // Text before description, per node, because the host walks the attributes in that
        // order within each node rather than making two passes over the subtree.
        for (CharSequence value : new CharSequence[]{node.getText(), node.getContentDescription()}) {
            if (value == null) {
                continue;
            }
            String trimmed = value.toString().trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            boolean duplicate = false;
            for (String already : seen) {
                if (already.equalsIgnoreCase(trimmed)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                seen.add(trimmed);
            }
        }
        for (int i = 0; i < node.getChildCount(); i++) {
            collectLabels(node.getChild(i), seen);
        }
    }

    /** Called on the accessibility callback; must stay cheap and must never throw. */
    void onEvent(AccessibilityEvent event) {
        if (!recording || steps.size() >= MAX_STEPS) {
            return;
        }
        snapshotIfDue(System.currentTimeMillis());
        try {
            String kind;
            switch (event.getEventType()) {
                case AccessibilityEvent.TYPE_VIEW_CLICKED:
                    kind = "tap";
                    break;
                case AccessibilityEvent.TYPE_VIEW_LONG_CLICKED:
                    kind = "long-press";
                    break;
                case AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED:
                    kind = "input";
                    break;
                case AccessibilityEvent.TYPE_VIEW_SCROLLED:
                    kind = "scroll";
                    break;
                case AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED: {
                    // Not an action — a consequence. If nothing was announced before it, the
                    // user moved the screen by some means the framework did not report, and
                    // the recording has a hole exactly here.
                    long now = System.currentTimeMillis();
                    boolean stillSettling = now - lastWindowChangeMs < WINDOW_SETTLE_MS;
                    lastWindowChangeMs = now;
                    if (stillSettling) {
                        return;   // the same navigation, still arriving
                    }
                    if (now - recordingStartedMs < ARMING_GRACE_MS) {
                        announcedSinceWindowChange = false;
                        return;   // the recorder arriving, not the user acting
                    }
                    if (!announcedSinceWindowChange) {
                        steps.add(new JSONObject()
                                .put("kind", "gap")
                                .put("reason", "screen_changed_with_no_announced_action")
                                .put("ts", now)
                                .put("package", event.getPackageName() == null
                                        ? "" : event.getPackageName().toString()));
                    }
                    announcedSinceWindowChange = false;
                    lastScrollSignature = "";
                    return;
                }
                default:
                    return;
            }
            announcedSinceWindowChange = true;

            JSONObject step = new JSONObject().put("kind", kind);
            if ("scroll".equals(kind)) {
                // A scroll fires continuously while a finger moves. Collapsing consecutive
                // scrolls of the same container keeps one replayable step instead of forty.
                String sig = String.valueOf(event.getPackageName()) + '/'
                        + String.valueOf(event.getClassName());
                if (sig.equals(lastScrollSignature)) {
                    return;
                }
                lastScrollSignature = sig;
                step.put("arg", "up");
            } else {
                lastScrollSignature = "";
            }

            AccessibilityNodeInfo source = event.getSource();
            if (source != null) {
                // For a press, address the node the HOST would have addressed. Its projection
                // folds an unlabelled, non-actionable child into its clickable ancestor, so a
                // selector recorded against the child would name an element that does not
                // exist in any analyze the host ever returns.
                AccessibilityNodeInfo target =
                        ("tap".equals(kind) || "long-press".equals(kind)) ? addressable(source) : source;
                String rid = target.getViewIdResourceName();
                if (rid != null && !rid.isEmpty()) {
                    // The host stores the tail, and matches on it.
                    int slash = rid.indexOf('/');
                    step.put("resource_id", slash >= 0 ? rid.substring(slash + 1) : rid);
                }
                CharSequence desc = target.getContentDescription();
                if (desc != null && desc.length() > 0) {
                    step.put("content_desc", desc.toString());
                }
                // A label is only durable for a control, not for the value of a text field —
                // and for `input` it would be the typed value itself.
                if (!"input".equals(kind)) {
                    CharSequence text = target.getText();
                    if (text != null && text.length() > 0) {
                        step.put("label", text.toString());
                    } else if ((desc == null || desc.length() == 0)
                            && ("tap".equals(kind) || "long-press".equals(kind))) {
                        // Presses only. A scroll is replayed by direction, so rolling its
                        // container up just staples the whole visible screen onto the step.
                        // Android list rows put the label on inner TextViews while the
                        // clickable element is the parent, so a row's own text is null and
                        // this recorded a tap with no selector at all — unreplayable, and the
                        // commonest shape in any settings-style screen. The host solves it by
                        // rolling the subtree's text up into the actionable node; recording
                        // anything else would name something `analyze` never reports.
                        String rolled = rolledUpLabel(target);
                        if (rolled != null) {
                            step.put("label", rolled);
                        }
                    }
                }
            }
            step.put("ts", System.currentTimeMillis());
            step.put("package", event.getPackageName() == null
                    ? "" : event.getPackageName().toString());
            steps.add(step);
        } catch (Exception e) {
            android.util.Log.w("AuaHelper", "dropped a recorded step: " + e);
        }
    }
}
