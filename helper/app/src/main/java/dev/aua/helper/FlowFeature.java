package dev.aua.helper;

import android.accessibilityservice.AccessibilityService;
import android.graphics.Rect;
import android.os.Bundle;
import android.view.accessibility.AccessibilityNodeInfo;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * {@code flow.run} — execute a run of AUA steps on the device.
 *
 * <p>AUA's host already models a goto route and a saved flow as the same thing: one list of
 * ``RouteStep``. Today the host drives that list one step at a time, paying a round trip and a
 * settle wait for each. Handing the whole list over means one round trip for the run, and each
 * step settles locally against the accessibility tree.
 *
 * <p>Only the UI subset is implemented, and that is deliberate. Roughly half of AUA's step
 * kinds are host operations — proxy, network shaping, feature flags, recursion into saved
 * routes — and a device cannot do them. Anything unrecognised stops the run and is reported
 * back by index, so the host resumes from exactly that step rather than the device guessing.
 */
final class FlowFeature implements Feature {

    /**
     * Kinds whose target is a predicate in ``arg``, matched on contains, as on the host.
     *
     * <p>``scroll-to`` belongs here too and was missed at first: it names its target the same
     * way, so leaving it out sent it down the element-selector path, where a step carrying
     * only ``arg`` matched nothing. The scrolling worked and the target was on screen — the
     * step still reported that it never appeared.
     */
    private static final java.util.Set<String> PREDICATE_KINDS = new java.util.HashSet<>(
            java.util.Arrays.asList(
                    "wait-for", "assert-visible", "assert-not-visible", "scroll-to"));

    /** How long a single step may wait for its target before giving up. */
    private static final long DEFAULT_STEP_TIMEOUT_MS = 5000L;

    /**
     * Checking steps take their budget from the step, not from this class.
     *
     * <p>The host gives {@code assert-visible} no wait at all — it asserts about the screen as
     * it is — while this ran the same step against a five-second poll. An element that turned
     * up 400ms late therefore passed here and would have failed there, and a device pass is
     * never re-run by the host, so the run carried on having "proved" something untrue. The
     * budget now travels on the step, and this is only the floor for a step that carries none.
     *
     * <p>Deliberately scoped to the predicate kinds. An acting step still needs its own find
     * wait: the device runs a whole flow without a re-analyze between steps, so a tap has to
     * be able to wait for the screen it is aiming at.
     */
    private static long checkBudget(String kind, JSONObject step, long runWide) {
        if (!PREDICATE_KINDS.contains(kind)) {
            return runWide;
        }
        return step.optLong("timeout_ms", runWide);
    }

    /** Polling gap while waiting for a node; the tree read is in-process and cheap. */
    private static final long FIND_INTERVAL_MS = 50L;

    private static final java.util.Set<String> DIRECTIONS = new java.util.HashSet<>(
            java.util.Arrays.asList("up", "down", "left", "right"));

    /** Upper bound on a scroll-to hunt, so a list that never ends cannot hang the run. */
    private static final int MAX_SCROLL_STEPS = 25;

    /** Quiet period that counts as "stable", matching the host's wait_stable. */
    private static final long STABLE_QUIET_MS = 600L;

    /** How long a step waits for the screen to react before moving on. */
    private static final long SETTLE_BUDGET_MS = 1500L;

    @Override
    public String namespace() {
        return "flow";
    }

    @Override
    public JSONObject handle(String method, JSONObject params) throws Exception {
        if (!"flow.run".equals(method)) {
            throw new IllegalArgumentException("unknown method: " + method);
        }
        JSONArray steps = params.optJSONArray("steps");
        if (steps == null) {
            throw new IllegalArgumentException("flow.run needs a `steps` array");
        }
        long stepTimeout = params.optLong("step_timeout_ms", DEFAULT_STEP_TIMEOUT_MS);

        JSONArray results = new JSONArray();
        int completed = 0;
        String stoppedReason = null;
        int stoppedAt = -1;

        for (int i = 0; i < steps.length(); i++) {
            JSONObject step = steps.getJSONObject(i);
            String kind = step.optString("kind", "");
            long began = System.currentTimeMillis();
            JSONObject r = new JSONObject().put("index", i).put("kind", kind);
            try {
                if (!runStep(kind, step, stepTimeout)) {
                    r.put("ok", false).put("error", "unsupported_kind");
                    results.put(r.put("ms", System.currentTimeMillis() - began));
                    stoppedReason = "unsupported_kind";
                    stoppedAt = i;
                    break;
                }
                r.put("ok", true);
            } catch (Exception e) {
                r.put("ok", false).put("error", e.getClass().getSimpleName() + ": " + e.getMessage());
                results.put(r.put("ms", System.currentTimeMillis() - began));
                stoppedReason = "step_failed";
                stoppedAt = i;
                break;
            }
            results.put(r.put("ms", System.currentTimeMillis() - began));
            completed++;
        }

        return new JSONObject()
                .put("completed", completed)
                .put("total", steps.length())
                .put("stopped_at", stoppedAt)
                .put("stopped_reason", stoppedReason == null ? JSONObject.NULL : stoppedReason)
                .put("steps", results);
    }

    /** @return false when this device cannot run the kind at all (host must take over). */
    private boolean runStep(String kind, JSONObject step, long timeoutMs) throws Exception {
        switch (kind) {
            case "tap": {
                AccessibilityNodeInfo n = require(step, timeoutMs);
                String before = signature();
                clickable(n).performAction(AccessibilityNodeInfo.ACTION_CLICK);
                settle(before, SETTLE_BUDGET_MS);
                return true;
            }
            case "long-press": {
                AccessibilityNodeInfo n = require(step, timeoutMs);
                String before = signature();
                clickable(n).performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK);
                settle(before, SETTLE_BUDGET_MS);
                return true;
            }
            case "input": {
                AccessibilityNodeInfo n = require(step, timeoutMs);
                Bundle args = new Bundle();
                args.putCharSequence(
                        AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                        step.optString("text", ""));
                n.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
                return true;
            }
            case "clear": {
                AccessibilityNodeInfo n = require(step, timeoutMs);
                Bundle args = new Bundle();
                args.putCharSequence(
                        AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "");
                n.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
                return true;
            }
            case "key": {
                Integer global = globalAction(step.optString("arg", ""));
                if (global == null) {
                    return false;   // an arbitrary keycode needs input injection, not a11y
                }
                String before = signature();
                service().performGlobalAction(global);
                settle(before, SETTLE_BUDGET_MS);
                return true;
            }
            case "tap-point": {
                int[] pt = parsePoint(step.optString("arg", ""));
                if (pt == null) {
                    return false;   // host reports unsupported_action for an unusable point
                }
                String before = signature();
                if (!Gestures.tap(service(), pt[0], pt[1])) {
                    throw new IllegalStateException("tap gesture was not delivered");
                }
                settle(before, SETTLE_BUDGET_MS);
                return true;
            }
            case "swipe": {
                String dir = step.optString("arg", "").toLowerCase();
                if (!DIRECTIONS.contains(dir)) {
                    return false;
                }
                String before = signature();
                if (!Gestures.swipe(service(), Gestures.screenBounds(service()), dir,
                        Gestures.DEFAULT_PERCENT)) {
                    throw new IllegalStateException("swipe gesture was not delivered");
                }
                settle(before, SETTLE_BUDGET_MS);
                return true;
            }
            case "scroll": {
                String dir = step.optString("arg", "").toLowerCase();
                if (!DIRECTIONS.contains(dir)) {
                    return false;
                }
                // The host treats a scroll that moved nothing as a failure on purpose:
                // "nothing left to scroll" and "my swipe missed the list" must not look the
                // same to a caller looping until something appears.
                if (!scrollOnce(dir)) {
                    throw new IllegalStateException("scroll moved nothing");
                }
                return true;
            }
            case "scroll-to": {
                String target = step.optString("arg", "");
                if (target.isEmpty()) {
                    return false;
                }
                // Host default is "up", which means "keep looking further down the list".
                String dir = step.optString("direction", "");
                if (dir.isEmpty()) {
                    dir = "up";
                }
                if (find(step, 0) != null) {
                    return true;   // already on screen; nothing to scroll
                }
                for (int n = 0; n < MAX_SCROLL_STEPS; n++) {
                    if (!scrollOnce(dir)) {
                        break;      // ran out of content
                    }
                    if (find(step, 0) != null) {
                        return true;
                    }
                }
                throw new IllegalStateException("scroll-to never revealed " + target);
            }
            case "hide-keyboard": {
                // setShowMode(SHOW_MODE_HIDDEN) is a standing preference, not a dismiss: it
                // returned success and left the keyboard on screen. Back does dismiss it, but
                // the host deliberately uses KEYCODE_ESCAPE instead because Back finishes the
                // Activity when no IME is up — and accessibility cannot send a raw keycode.
                // Checking for the input-method window first makes Back safe: if one is
                // showing, Back is guaranteed to consume it rather than the screen.
                if (!imeWindowShowing()) {
                    return true;   // already hidden; nothing to do
                }
                String before = signature();
                service().performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                settle(before, SETTLE_BUDGET_MS);
                if (imeWindowShowing()) {
                    throw new IllegalStateException("the keyboard is still showing");
                }
                return true;
            }
            case "paste": {
                AccessibilityNodeInfo focused =
                        service().findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
                if (focused == null) {
                    throw new IllegalStateException("nothing is focused to paste into");
                }
                String before = signature();
                focused.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                settle(before, SETTLE_BUDGET_MS);
                return true;
            }
            case "wait-stable": {
                long budget = step.optLong("timeout_ms", 15000L);
                if (!waitStable(STABLE_QUIET_MS, budget)) {
                    throw new IllegalStateException("screen never went quiet");
                }
                return true;
            }
            case "wait-for": {
                if (find(step, checkBudget(kind, step, timeoutMs)) == null) {
                    throw new IllegalStateException("wait-for target never appeared");
                }
                return true;
            }
            case "assert-visible": {
                if (find(step, checkBudget(kind, step, timeoutMs)) == null) {
                    throw new IllegalStateException("assert-visible target not present");
                }
                return true;
            }
            case "assert-not-visible": {
                if (find(step, 0) != null) {
                    throw new IllegalStateException("assert-not-visible target is present");
                }
                return true;
            }
            default:
                return false;
        }
    }

    /**
     * One scroll of the scrollable container, reporting whether the screen actually moved.
     *
     * <p>Movement is checked rather than assumed because the host draws a hard line between
     * "there is nothing left to scroll" and "my swipe missed the list". Returning true for
     * both would let a caller loop forever waiting for content that a mis-aimed gesture was
     * never going to reveal.
     */
    private boolean scrollOnce(String direction) {
        AccessibilityService s = service();
        Rect box = scrollableBox(s);
        String before = signature();
        if (!Gestures.swipe(s, box, direction, Gestures.DEFAULT_PERCENT)) {
            throw new IllegalStateException("scroll gesture was not delivered");
        }
        settle(before, SETTLE_BUDGET_MS);
        return !signature().equals(before);
    }

    /**
     * The biggest scrollable node, or the whole screen when nothing advertises itself.
     *
     * <p>Aiming at the container rather than the display matters on screens where the list
     * occupies part of the window: a full-screen swipe can start on a toolbar or a bottom bar
     * and scroll nothing at all.
     */
    private Rect scrollableBox(AccessibilityService s) {
        AccessibilityNodeInfo root = s.getRootInActiveWindow();
        Rect best = null;
        if (root != null) {
            best = largestScrollable(root, null);
        }
        return best != null ? best : Gestures.screenBounds(s);
    }

    private static Rect largestScrollable(AccessibilityNodeInfo node, Rect best) {
        Rect winner = best;
        if (node.isScrollable()) {
            Rect b = new Rect();
            node.getBoundsInScreen(b);
            long area = (long) b.width() * b.height();
            if (winner == null || area > (long) winner.width() * winner.height()) {
                winner = b;
            }
        }
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                winner = largestScrollable(child, winner);
            }
        }
        return winner;
    }

    /** Is an input-method window on screen right now? */
    private boolean imeWindowShowing() {
        AccessibilityService s = service();
        for (android.view.accessibility.AccessibilityWindowInfo w : s.getWindows()) {
            if (w.getType() == android.view.accessibility.AccessibilityWindowInfo
                    .TYPE_INPUT_METHOD) {
                return true;
            }
        }
        return false;
    }

    /** Wait until the screen holds the same signature for *quietMs*, or give up at *budgetMs*. */
    private boolean waitStable(long quietMs, long budgetMs) {
        long deadline = System.currentTimeMillis() + budgetMs;
        String last = signature();
        long since = System.currentTimeMillis();
        while (System.currentTimeMillis() < deadline) {
            try {
                Thread.sleep(FIND_INTERVAL_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
            String now = signature();
            if (now.equals(last)) {
                if (System.currentTimeMillis() - since >= quietMs) {
                    return true;
                }
            } else {
                last = now;
                since = System.currentTimeMillis();
            }
        }
        return false;
    }

    /** ``"x,y"`` -> {x, y}; null when it is not a usable pair, exactly as the host parses it. */
    private static int[] parsePoint(String arg) {
        if (arg == null || arg.isEmpty()) {
            return null;
        }
        String[] parts = arg.replace(" ", "").split(",");
        if (parts.length != 2) {
            return null;
        }
        try {
            int x = Math.round(Float.parseFloat(parts[0]));
            int y = Math.round(Float.parseFloat(parts[1]));
            return (x >= 0 && y >= 0) ? new int[]{x, y} : null;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * A cheap signature of what is on screen: enough to notice a navigation, not a caret blink.
     */
    private String signature() {
        AccessibilityService s = HelperService.awaitService(2500L);
        AccessibilityNodeInfo root = s == null ? null : s.getRootInActiveWindow();
        if (root == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        collect(root, sb, new int[]{400});
        return sb.toString();
    }

    private static void collect(AccessibilityNodeInfo n, StringBuilder sb, int[] budget) {
        if (budget[0]-- <= 0) {
            return;
        }
        CharSequence t = n.getText();
        if (t != null && t.length() > 0) {
            sb.append(t).append('\u0001');
        }
        for (int i = 0; i < n.getChildCount(); i++) {
            AccessibilityNodeInfo c = n.getChild(i);
            if (c != null) {
                collect(c, sb, budget);
            }
        }
    }

    /**
     * Wait for the screen to actually respond to the step just performed.
     *
     * <p>Without this the run is fast and wrong. {@code performGlobalAction} returns
     * immediately, so a Back followed by a check matched the *previous* screen's nodes and
     * every step reported success while the flow had not navigated at all — 8/8 passed on a
     * run that never left the second page. Correctness first: a step is not done until the
     * screen it acted on has changed, or the budget says it never will.
     */
    private void settle(String before, long budgetMs) {
        long deadline = System.currentTimeMillis() + budgetMs;
        while (System.currentTimeMillis() < deadline) {
            if (!signature().equals(before)) {
                return;
            }
            try {
                Thread.sleep(FIND_INTERVAL_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    /** Only the global keys map onto accessibility; the rest belong to the host. */
    private static Integer globalAction(String name) {
        switch (name == null ? "" : name.toLowerCase()) {
            case "back":
                return AccessibilityService.GLOBAL_ACTION_BACK;
            case "home":
                return AccessibilityService.GLOBAL_ACTION_HOME;
            case "recents":
            case "recent":
                return AccessibilityService.GLOBAL_ACTION_RECENTS;
            default:
                return null;
        }
    }

    /**
     * Walk up to a node that can actually take the click.
     *
     * <p>AUA's selectors point at the labelled node, which is very often a TextView inside a
     * clickable row. Calling ACTION_CLICK on the label is silently a no-op, so the run would
     * report success while the screen never moved.
     */
    private static AccessibilityNodeInfo clickable(AccessibilityNodeInfo node) {
        AccessibilityNodeInfo n = node;
        for (int depth = 0; n != null && depth < 8; depth++) {
            if (n.isClickable()) {
                return n;
            }
            n = n.getParent();
        }
        return node;
    }

    private AccessibilityNodeInfo require(JSONObject step, long timeoutMs) throws Exception {
        AccessibilityNodeInfo n = find(step, timeoutMs);
        if (n == null) {
            throw new IllegalStateException("no node matched " + selectorOf(step));
        }
        return n;
    }

    private static String selectorOf(JSONObject step) {
        return "rid=" + step.optString("resource_id", "")
                + " label=" + step.optString("label", "")
                + " desc=" + step.optString("content_desc", "");
    }

    private AccessibilityService service() {
        AccessibilityService s = HelperService.awaitService(2500L);
        if (s == null) {
            throw new IllegalStateException("accessibility service is not attached");
        }
        return s;
    }

    private AccessibilityNodeInfo find(JSONObject step, long timeoutMs) {
        long deadline = System.currentTimeMillis() + Math.max(0L, timeoutMs);
        while (true) {
            AccessibilityService s = HelperService.awaitService(2500L);
            AccessibilityNodeInfo root = s == null ? null : s.getRootInActiveWindow();
            if (root != null) {
                AccessibilityNodeInfo hit = match(root, step);
                if (hit != null) {
                    return hit;
                }
            }
            if (System.currentTimeMillis() >= deadline) {
                return null;
            }
            try {
                Thread.sleep(FIND_INTERVAL_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
    }

    /**
     * Selector order mirrors the host's: resource id first, then label, then description.
     *
     * <p>``wait-for`` and the asserts are different, and must stay different: on the host they
     * take a predicate in ``arg`` with a ``by`` field, and they match on *contains*, not
     * equality. Matching those on ``label`` with equality made the same step mean two
     * different things depending on where it ran, which is precisely what this path is not
     * allowed to do.
     */
    private static AccessibilityNodeInfo match(AccessibilityNodeInfo node, JSONObject step) {
        String arg = step.optString("arg", "");
        String kind = step.optString("kind", "");
        if (!arg.isEmpty() && PREDICATE_KINDS.contains(kind)) {
            if (matchesPredicate(node, step.optString("by", "text"), arg) && onScreen(node)) {
                return node;
            }
            return descend(node, step);
        }

        String rid = step.optString("resource_id", "");
        String label = step.optString("label", "");
        String desc = step.optString("content_desc", "");

        if (!rid.isEmpty()) {
            String full = node.getViewIdResourceName();
            // The host stores the id TAIL, the device reports "package:id/name".
            if (full != null && (full.equals(rid) || full.endsWith("/" + rid)) && onScreen(node)) {
                return node;
            }
        } else if (!label.isEmpty()) {
            CharSequence text = node.getText();
            if (text != null && text.toString().equals(label) && onScreen(node)) {
                return node;
            }
        } else if (!desc.isEmpty()) {
            CharSequence cd = node.getContentDescription();
            if (cd != null && cd.toString().equals(desc) && onScreen(node)) {
                return node;
            }
        }

        return descend(node, step);
    }

    /**
     * Does this node satisfy a ``by``-qualified predicate, using the host's vocabulary?
     *
     * <p>The two sides had different vocabularies, which is worse than it sounds because the
     * mismatch was silent in both directions. ``id`` — the spelling the flow parser actually
     * emits for a resource-id predicate — fell through the chain to {@code getText()}, so the
     * device searched a screen's *labels* for a resource id, found nothing, and reported the
     * element absent. Meanwhile ``content_desc`` and ``resource_id`` were accepted here but
     * are hard errors on the host, so a step could run on the device and then be a usage
     * error the instant the host touched it.
     *
     * <p>``text`` searches the description too, because {@code _BY_FIELDS["text"]} on the host
     * is {@code ["text", "description"]}. Matching only the text made {@code assert-not-visible}
     * pass on a described-but-untexted node the host would plainly have found — a false pass,
     * and false passes are the ones that survive.
     */
    private static boolean matchesPredicate(AccessibilityNodeInfo node, String by, String arg) {
        switch (by == null || by.isEmpty() ? "text" : by) {
            case "desc":
                return contains(node.getContentDescription(), arg);
            case "rid":
            case "id":
                return contains(node.getViewIdResourceName(), arg);
            case "text":
                return contains(node.getText(), arg) || contains(node.getContentDescription(), arg);
            default:
                // Refusing beats degrading. A fall-through to a text search is how the host
                // once spent a full timeout hunting a resource id among the labels and then
                // reported the screen wrong — a claim about the product, not about the query.
                throw new IllegalArgumentException("unknown selector field 'by=" + by + "'");
        }
    }

    private static boolean contains(CharSequence haystack, String needle) {
        return haystack != null && haystack.toString().contains(needle);
    }

    /**
     * Is this node where the host's projection would still be able to see it?
     *
     * <p>The host does not assert about the raw tree. ``parse_hierarchy`` drops any node with
     * degenerate bounds or bounds entirely off the display, so its element list — the thing
     * every host-side check runs against — simply does not contain them. This walked the raw
     * tree instead, and a recycled list row that keeps its resource id while parked off-screen
     * therefore passed {@code assert-visible} here and failed it there.
     *
     * <p>Deliberately bounds, not {@code isVisibleToUser()}: the host's filter is geometric and
     * knows nothing about occlusion, so testing occlusion here would diverge the other way.
     */
    private static boolean onScreen(AccessibilityNodeInfo node) {
        Rect box = new Rect();
        node.getBoundsInScreen(box);
        if (box.width() <= 0 || box.height() <= 0) {
            return false;
        }
        AccessibilityService s = HelperService.awaitService(2500L);
        if (s == null) {
            return true;   // cannot tell; do not invent a reason to reject
        }
        Rect screen = Gestures.screenBounds(s);
        return !(box.right <= 0 || box.bottom <= 0
                || box.left >= screen.width() || box.top >= screen.height());
    }

    private static AccessibilityNodeInfo descend(AccessibilityNodeInfo node, JSONObject step) {
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo hit = match(child, step);
                if (hit != null) {
                    return hit;
                }
            }
        }
        return null;
    }
}
