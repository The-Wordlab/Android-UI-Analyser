package dev.aua.helper;

import android.view.accessibility.AccessibilityNodeInfo;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A live accessibility tree reduced to the handful of nodes that carry a decision.
 *
 * <p>Measured against 638 screens harvested from live emulators — stock AOSP, third-party apps, and
 * the same screens under English, Arabic and German. A real screen has a median of 41 nodes and
 * about 6 that can be acted on; this keeps the actionable ones and at most two lines of context.
 *
 * <p>Three facts from that harvest shaped this, and each contradicted an earlier assumption:
 *
 * <ul>
 *   <li><b>Text is the common field, not the rare one.</b> 61.8% of clickable nodes carry visible
 *       text, 32.0% a description, 34.7% a resource id. An earlier design preferred the resource id
 *       and got 7.6% coverage on live Settings screens.
 *   <li><b>11.2% of clickable nodes carry neither text nor description.</b> No selector can name
 *       those. They stay in the projection and remain addressable by position, which is the whole
 *       reason this hands the caller an index rather than a string.
 *   <li><b>25% of clickable texts are duplicated elsewhere on their own screen.</b> Android puts a
 *       heading above a row with the same words. That is layout, not a puzzle, so it is resolved
 *       here rather than pushed to whatever does the choosing.
 * </ul>
 *
 * <p>What this deliberately cannot fix: a drawn surface does not produce badly-named nodes, it
 * produces <em>one node or none</em>. A live backgammon board reports six clickable nodes, all
 * named, all of them left-column chrome — the board itself contributes nothing. A 2048 board
 * reports zero. On such a screen the honest answer is to hand off, and this makes that visible by
 * returning an empty or near-empty list rather than inventing a choice.
 *
 * <p>Mirrors {@code experiments/functiongemma/v11_projection.py}. That file is the reference and
 * carries the measurements; this is the on-device implementation, and the two must agree.
 */
final class Projection {

    /**
     * Nodes offered to the decider. Raised from 14 after the first live drive on a real app failed
     * on it.
     *
     * <p>14 was a model-era number: it existed to keep a language model's prompt small. What decides
     * here is a scoring rule that walks the list once, so nodes cost nothing to show.
     *
     * <p>The live failure generalises. Asked to reach an app's Apps section, this scrolled three
     * times and reported {@code target_absent} while an "Apps" control sat plainly on screen. A fixed
     * bottom navigation bar is <em>last in tree order and first in importance</em>: the list above it
     * consumed all 14 slots, and because the bar is not inside the scrollable list, scrolling could
     * never reveal it. The refusal was internally consistent and factually wrong.
     *
     * <p>Over the 638 harvested screens, actionable nodes per screen are p50 7, p90 17, p95 22. A cap
     * of 14 truncates 13.2% of screens, 22 truncates 4.4%, 28 truncates 2.0%, and 32 also truncates
     * 2.0% — so 28 is where more stops buying anything. The remaining 2% carry 48 to 72 actionable
     * nodes, which no reachable cap fixes and which {@code more} reports honestly.
     */
    static final int MAX_NODES = 28;

    /** Context lines kept — normally the screen's heading. Capped so a status bar cannot crowd
     * out a control: actionable nodes are ranked first and never displaced. */
    static final int MAX_CONTEXT = 2;

    /** Status-bar and shade content that survives the structural filter by carrying text: the
     * clock, the battery, signal strength, notification summaries. Useless for deciding what to
     * tap, and notification text belongs to other applications. */
    private static final String[] SYSTEM_RID_PREFIXES = {
        "status_bar", "statusIcons", "notification", "clock", "battery",
        "wifi_", "mobile_", "system_icons", "shade", "qs_", "keyguard",
    };

    /** One node as the decider sees it. Holds the live node so whatever chooses can act on it
     * directly — no selector string is ever produced, so none can be wrong. */
    static final class Item {
        final int index;                    // 1-based; "n3" to a caller
        final AccessibilityNodeInfo node;
        final String text;
        final String desc;
        final String rid;
        final boolean tappable;
        final boolean scrollable;

        Item(int index, AccessibilityNodeInfo node, String text, String desc, String rid,
                boolean tappable, boolean scrollable) {
            this.index = index;
            this.node = node;
            this.text = text;
            this.desc = desc;
            this.rid = rid;
            this.tappable = tappable;
            this.scrollable = scrollable;
        }

        /** Text and description together — what the decider scores a goal against. */
        String label() {
            if (text.isEmpty()) {
                return desc;
            }
            if (desc.isEmpty() || desc.equals(text)) {
                return text;
            }
            return text + " " + desc;
        }
    }

    final List<Item> items;
    /** True when actionable nodes were dropped for space. This changes what the right answer
     * <em>is</em>: with content off screen, "the target is absent" is not provable and scrolling is
     * the only sound step. A caller that ignores this will learn to refuse on any long list. */
    final boolean more;

    private Projection(List<Item> items, boolean more) {
        this.items = items;
        this.more = more;
    }

    /** Flatten, filter, collapse duplicates and number what is left. */
    static Projection of(AccessibilityNodeInfo root) {
        List<AccessibilityNodeInfo> flat = new ArrayList<>();
        flatten(root, flat, 0);

        List<AccessibilityNodeInfo> kept = new ArrayList<>();
        for (AccessibilityNodeInfo node : flat) {
            if (!isNoise(node)) {
                kept.add(node);
            }
        }

        // Text that some actionable node already owns. A non-actionable twin adds nothing.
        Set<String> actionableText = new LinkedHashSet<>();
        for (AccessibilityNodeInfo node : kept) {
            if (node.isClickable()) {
                String t = normalise(node.getText());
                if (!t.isEmpty()) {
                    actionableText.add(t);
                }
            }
        }

        List<AccessibilityNodeInfo> actionable = new ArrayList<>();
        List<AccessibilityNodeInfo> context = new ArrayList<>();
        for (AccessibilityNodeInfo node : kept) {
            boolean act = node.isClickable() || node.isScrollable();
            if (act) {
                actionable.add(node);
                continue;
            }
            String t = normalise(node.getText());
            if (!t.isEmpty() && actionableText.contains(t)) {
                continue;   // the duplicated heading
            }
            context.add(node);
        }

        boolean truncated = actionable.size() > MAX_NODES;
        List<Item> out = new ArrayList<>();
        int n = 1;
        for (int i = 0; i < actionable.size() && n <= MAX_NODES; i++, n++) {
            out.add(item(n, actionable.get(i)));
        }
        // A heading carries text; a notification summary carries only a description. Preferring
        // text keeps the two context slots for orientation rather than for other apps' chatter.
        int room = Math.min(MAX_NODES - out.size(), MAX_CONTEXT);
        for (int pass = 0; pass < 2 && room > 0; pass++) {
            for (int i = 0; i < context.size() && room > 0; i++) {
                AccessibilityNodeInfo c = context.get(i);
                boolean hasText = !normalise(c.getText()).isEmpty();
                if (hasText == (pass == 0)) {
                    out.add(item(n++, c));
                    room--;
                }
            }
        }
        return new Projection(out, truncated);
    }

    /**
     * Text and description belonging to a node, folded up from its own subtree when it has none.
     *
     * <p>This is the difference between working and not working on a real device, and it cost a
     * whole driving run to find. In AOSP Settings the <em>clickable</em> row is an empty container:
     * on a live Settings home screen 19 of its 21 clickable nodes carry no text, no description and
     * no resource id of their own, and "Display" appears on none of them. The label sits on a
     * {@code android:id/title} child with {@code android:id/summary} beside it.
     *
     * <p>AUA's host-side projection already folds those children into the row, which is why
     * {@code aua analyze} reports one node reading "Network &amp; internet Mobile, Wi-Fi, hotspot"
     * and why every measurement taken through it looked healthy while this class saw blanks. Node
     * for node, the two were not looking at the same thing.
     *
     * <p>Bounded: only the first few descendants, only until something is found, so a list row
     * cannot absorb the whole list.
     */
    private static String folded(AccessibilityNodeInfo node, boolean wantText, int depth) {
        if (node == null || depth > 4) {
            return "";
        }
        String own = normalise(wantText ? node.getText() : node.getContentDescription());
        if (!own.isEmpty()) {
            return own;
        }
        StringBuilder out = new StringBuilder();
        int count = Math.min(node.getChildCount(), 8);
        for (int i = 0; i < count; i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child == null || child.isClickable()) {
                // A clickable child owns its own label; folding it up would steal it.
                continue;
            }
            String part = folded(child, wantText, depth + 1);
            if (part.isEmpty()) {
                continue;
            }
            if (out.length() > 0) {
                out.append(' ');
            }
            out.append(part);
            if (out.length() > 160) {
                break;
            }
        }
        return out.toString();
    }

    private static Item item(int n, AccessibilityNodeInfo node) {
        // Fold only for actionable nodes: a heading already carries its own text, and folding a
        // container would invent labels for scaffolding.
        boolean act = node.isClickable() || node.isScrollable();
        String text = act ? folded(node, true, 0) : normalise(node.getText());
        String desc = act ? folded(node, false, 0) : normalise(node.getContentDescription());
        String rid = ridTail(node.getViewIdResourceName());
        // The id is shown only when nothing else names the node. Showing it everywhere is how an
        // earlier corpus taught a model that the id was just the label in another font.
        if (!text.isEmpty() || !desc.isEmpty()) {
            rid = "";
        }
        return new Item(n, node, text, desc, rid, node.isClickable(), node.isScrollable());
    }

    private static void flatten(AccessibilityNodeInfo node, List<AccessibilityNodeInfo> out, int depth) {
        if (node == null || depth > 40 || out.size() > 600) {
            return;
        }
        out.add(node);
        for (int i = 0; i < node.getChildCount(); i++) {
            flatten(node.getChild(i), out, depth + 1);
        }
    }

    /**
     * A node that can neither be acted on nor read, or that belongs to the system bars.
     *
     * <p>Deliberately not a list of known container ids. The reference implementation started that
     * way and leaked {@code status_bar_launch_animation_container} and every other wrapper nobody
     * had thought to name: an enumerated list only ever covers what someone remembered.
     */
    private static boolean isNoise(AccessibilityNodeInfo node) {
        if (node == null) {
            return true;
        }
        boolean act = node.isClickable() || node.isScrollable();
        String text = act ? folded(node, true, 0) : normalise(node.getText());
        String desc = act ? folded(node, false, 0) : normalise(node.getContentDescription());
        if (!act && text.isEmpty() && desc.isEmpty()) {
            return true;
        }
        if (!act) {
            String rid = ridTail(node.getViewIdResourceName());
            for (String prefix : SYSTEM_RID_PREFIXES) {
                if (rid.startsWith(prefix)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static String ridTail(CharSequence rid) {
        if (rid == null) {
            return "";
        }
        String s = rid.toString();
        int slash = s.lastIndexOf('/');
        return slash >= 0 ? s.substring(slash + 1) : s;
    }

    /**
     * NFKC, non-breaking punctuation flattened, whitespace collapsed.
     *
     * <p>Real screens contain U+2011 in "Wi&#8209;Fi" and U+00A0 between words; matching should not
     * turn on which dash a designer used.
     */
    static String normalise(CharSequence value) {
        if (value == null) {
            return "";
        }
        String s = Normalizer.normalize(value.toString(), Normalizer.Form.NFKC);
        StringBuilder out = new StringBuilder(s.length());
        boolean space = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '‑': case '‐': case '–': case '—': c = '-'; break;
                case '‘': case '’': c = '\''; break;
                case '“': case '”': c = '"'; break;
                case '​': continue;
                case ' ': c = ' '; break;
                default: break;
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                space = out.length() > 0;
                continue;
            }
            if (space) {
                out.append(' ');
                space = false;
            }
            out.append(c);
        }
        return out.toString();
    }

    /** Lowercased alphanumeric runs. No regex: this runs on every node of every frame. */
    static List<String> words(String value) {
        List<String> out = new ArrayList<>();
        if (value == null || value.isEmpty()) {
            return out;
        }
        String lower = value.toLowerCase(Locale.ROOT);
        StringBuilder token = new StringBuilder();
        for (int i = 0; i < lower.length(); i++) {
            char c = lower.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                token.append(c);
            } else if (token.length() > 0) {
                out.add(token.toString());
                token.setLength(0);
            }
        }
        if (token.length() > 0) {
            out.add(token.toString());
        }
        return out;
    }
}
