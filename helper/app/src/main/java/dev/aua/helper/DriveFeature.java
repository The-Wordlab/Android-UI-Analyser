package dev.aua.helper;

import android.accessibilityservice.AccessibilityService;
import android.view.accessibility.AccessibilityNodeInfo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.Set;

/**
 * {@code drive.run} — take a goal and reach it, deciding each step on the device.
 *
 * <p>Until now the helper replayed: {@code flow.run} takes a {@code steps} array the host already
 * planned. This decides instead, which is what makes the loop autonomous — observe, choose, act,
 * observe — with no host round trip and no model.
 *
 * <p><b>Why a scoring rule and not a language model.</b> On 26 goals against a live Android 16
 * device, a fine-tuned 350M policy reached <b>0 of 19</b> reachable destinations and authored a
 * selector that existed on screen <b>0 times out of 71</b>; every one of its 16 checkpoints scored
 * zero on grounding. This rule reached <b>17 of 19</b>. The reason is structural rather than a
 * matter of quality: this picks an element out of the list it was shown, so it cannot name one that
 * does not exist, while a model that spells selector strings can and did — it emitted destinations
 * from its training corpus.
 *
 * <p>The margin also shows how little of this task is difficult. Flat word overlap taps
 * "Search settings" for any goal containing the word "settings" and reaches only 9 of 19; adding a
 * first-token bonus and a score floor reaches 17. Tie-breaking, not understanding, decides most of
 * it.
 *
 * <p><b>The two things this could not do</b> are now done, and neither needed a model.
 *
 * <ul>
 *   <li><b>Recognise the impossible.</b> Four of seven goals needing a host capability ended stuck,
 *       because "nothing scored well" read the same as "keep scrolling". {@link #HOST_TERMS} refuses
 *       those from the goal text before a node is scored. The list is short because the admission
 *       rule is strict in the expensive direction: see that field.
 *   <li><b>Bridge a semantic gap.</b> Two of nineteen were missed where "the words simply do not
 *       meet". They do meet — the tokeniser was dropping the joint. "Internet AndroidWifi" reduces
 *       to {@code [internet, androidwifi]} and "Reset Bluetooth &amp; Wi-Fi" to {@code [..., wi,
 *       fi]}, so a goal saying {@code wifi} scored zero against both. {@link #variants} reads the
 *       joint in each direction. Capitalisation and punctuation, not meaning.
 * </ul>
 *
 * <p><b>And a third, which was a defect rather than a gap.</b> After a tap that did not navigate,
 * this re-scored the settled screen, picked the same row, and pressed it again until the budget ran
 * out — in a log that reads as a hung helper, not as a rule with nothing left to try. The run now
 * counts attempts per node ({@code tried}) and what came of each ({@code last}), the field shape
 * {@code experiments/functiongemma/v12_progress.py} defines, and refuses {@code no_progress} when
 * the node the goal names is the one that stalled. A stalled node <em>elsewhere</em> still means
 * nothing: those screens are tapped, not refused.
 *
 * <p>What remains is a true synonym — "make the text bigger" for "Display" — which is derivable
 * from neither string. There is nowhere on the device to keep one: this class's word tables are
 * {@code private static final}, the helper has no method through which the host can seed state, and
 * the wire protocol has no field for it. That gap stays open.
 *
 * <p>Mirrors {@code src/android_ui_analyser/drive_rule.py}, whose tests are this class's
 * specification; the two must agree. Scrolling and key presses are delegated to
 * {@link FlowFeature} so settle, gesture dispatch and movement detection have one implementation.
 * A tap acts on the chosen {@link AccessibilityNodeInfo} directly — the projection already holds
 * it, so there is no selector to get wrong, and the duplicate-label problem that costs host-side
 * drivers cannot arise.
 */
final class DriveFeature implements Feature {

    /** Words that name no destination. Short on purpose: a long list starts encoding assumptions
     * about phrasing, and we do not control how goals are worded. */
    private static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
            "a", "an", "and", "the", "to", "of", "on", "in", "at", "for", "with", "from",
            "me", "my", "i", "you", "it", "its", "this", "that", "these", "those",
            "is", "are", "be", "was", "were", "do", "does", "did", "can", "could",
            "would", "should", "will", "please", "just", "then", "now", "up",
            "open", "go", "goto", "show", "find", "get", "reach", "take", "bring",
            "land", "navigate", "screen", "page", "section", "settings", "setting",
            "view", "tab", "prove", "confirm", "need", "want", "let", "make"));

    /** Chrome present on nearly every screen and almost never the destination. Scoring it normally
     * is exactly how flat overlap lost eight of its nineteen. */
    private static final Set<String> CHROME = new HashSet<>(Arrays.asList(
            "search settings", "search", "navigate up", "back", "more options", "options",
            "home", "close", "cancel", "dismiss", "open features menu", "overflow"));

    /**
     * Words naming something only the host can do. A goal containing one is refused before any node
     * is scored, because no amount of tapping reaches {@code aua screenshot}.
     *
     * <p>Two tests admit a word, and the second is why this list is short. It must name a capability
     * the {@code aua} CLI has, <em>and</em> it must almost never appear in an on-screen label —
     * every word here appears in at most 1% of the 638 real screens under
     * {@code runs/functiongemma/screens}, and most in none of them.
     *
     * <p>The second test has to be made against screens somebody else's apps produced; checking a
     * refusal list against a generated list of host goals proves only that one hand wrote both.
     * Measured against the harvest, the words a <em>complete</em> list would need are exactly the
     * ones that cannot be allowed: "system" is in 83% of real screens, "screen" 53%, "network" 3.4%,
     * "clock" and "time" 4-7%. So "change the system time" and "turn the network off" are not
     * refusable from goal text, and this does not try. The collision is structural — AUA's host
     * capabilities are named after the same device concepts Settings exposes as destinations.
     *
     * <p>The trade leans hard towards precision because the errors are not comparable. A missed
     * refusal costs a vaguer handoff: budget is spent, nothing matches, the run hands off as
     * {@code target_absent} instead of {@code needs_host}, and the host gets control either way. A
     * false refusal stops a run that would have succeeded, and nothing later recovers it.
     *
     * <p>Mirrors {@code drive_rule.HOST_TERMS}, which records the two words measurement removed.
     */
    private static final Set<String> HOST_TERMS = new HashSet<>(Arrays.asList(
            "adb", "apk", "capture", "database", "dump", "emulator", "host", "install",
            "landscape", "logcat", "offline", "proxy", "query", "recording", "restore",
            "rotate", "screenshot", "sqlite", "traffic"));

    /**
     * The decline control of a consent dialog — the only screen evidence that going on would mean
     * granting something. Used for nothing but naming a refusal that was already happening.
     *
     * <p>Three labels and not six, on purpose. "Cancel" is already chrome, and "No thanks" / "Not
     * now" are the buttons on rating prompts and update nags, which gate nothing.
     */
    private static final Set<String> DECLINE_LABELS = new HashSet<>(Arrays.asList(
            "deny", "don't allow", "dont allow"));

    /**
     * {@code last} values meaning the run is not advancing. Mirrors {@code v12_progress.STALLED}.
     */
    /**
     * Visibility phrases that end a goal asking a <em>question</em> about the screen rather than
     * giving it an instruction. Longest first: "is showing" must win over "showing".
     *
     * <p>Mirrors {@code drive_rule.LOOK_TAILS}. Over 51,658 corpus rows these plus
     * {@link #LOOK_HEADS} identify 100% of the {@code done} rows and fire on 0.1-0.5% of every
     * acting class. Before this existed the rule answered {@code done} correctly 7.6% of the time.
     */
    private static final String[] LOOK_TAILS = {
        "can be seen", "is displayed", "is showing", "is visible", "is here",
        "on screen", "be seen", "displayed", "showing", "visible",
    };

    /**
     * Openings that make a goal a question on their own. Deliberately narrow: the corpus opens 515
     * look-goals <em>and</em> 2,645 acting goals with "can you", so only the verb can decide it.
     */
    private static final String[] LOOK_HEADS = {
        "can you tell me if ", "can you confirm ", "can you verify ", "do you see ",
        "can you see ", "can i see ", "let me see ", "tell me if ",
    };

    /** Question verbs stripped off a goal already identified by a {@link #LOOK_TAILS} phrase. */
    private static final String[] LOOK_LEADS = {
        "please confirm ", "please check ", "make sure ", "confirm that ", "verify that ",
        "check that ", "confirm ", "verify ", "ensure ", "assert ", "check ", "does ",
        "are ", "is ", "do ",
    };

    private static final Set<String> STALLED = new HashSet<>(Arrays.asList("blocked", "unchanged"));

    /** An action landed and the screen became a different screen. */
    private static final String CHANGED = "changed";
    /** An action landed and nothing happened. The evidence for {@code no_progress}. */
    private static final String UNCHANGED = "unchanged";
    /** The node refused the click outright. Also evidence. */
    private static final String BLOCKED = "blocked";

    private static final double SCORE_FLOOR = 1.0;
    private static final double FIRST_TOKEN_BONUS = 1.5;
    private static final double WORD_MATCH = 1.0;
    private static final double RID_MATCH = 0.5;

    private static final int DEFAULT_BUDGET = 8;
    private static final int MAX_SCROLLS = 3;

    private final FlowFeature flow;

    DriveFeature(FlowFeature flow) {
        this.flow = flow;
    }

    @Override
    public String namespace() {
        return "drive";
    }

    @Override
    public JSONObject handle(String method, JSONObject params) throws Exception {
        if (!"drive.run".equals(method)) {
            throw new IllegalArgumentException("unknown method: " + method);
        }
        String goal = params.optString("goal", "");
        if (goal.trim().isEmpty()) {
            throw new IllegalArgumentException("drive.run needs a `goal`");
        }
        int budget = Math.max(1, params.optInt("budget", DEFAULT_BUDGET));

        List<String> terms = terms(goal);
        JSONArray steps = new JSONArray();
        int scrolls = 0;
        String stop = "budget_exhausted";

        // Per run, not per instance: this Feature is registered once and serves every goal, so a
        // field here would carry one goal's attempts into the next. Keyed by rid + label rather
        // than by "n3", because the ordinal is positional and every re-projection renumbers.
        Map<String, Integer> tried = new HashMap<>();
        Map<String, String> last = new HashMap<>();

        // Before the screen is even read, not after. A host goal is unreachable on every screen, so
        // looking first only decides how much budget gets spent proving it.
        String host = hostTerm(terms);
        if (host != null) {
            JSONObject refusal = new JSONObject();
            refusal.put("step", 0);
            refusal.put("decision", "handoff");
            refusal.put("reason", "needs_host");
            refusal.put("term", host);
            steps.put(refusal);
            JSONObject early = new JSONObject();
            early.put("goal", goal);
            early.put("stop_reason", "handoff");
            early.put("steps", steps);
            early.put("step_count", steps.length());
            return early;
        }

        // A goal that only asks what is on screen. Every node is eligible, nothing is pressed, and
        // an absent subject falls through to the same scroll-then-refuse path every other goal
        // uses — `target_absent` being the honest answer to "is it visible" when it is not.
        String subject = onlyAsksToLook(goal);
        boolean lookOnly = subject != null;
        if (lookOnly) {
            terms = terms(subject);
        }

        for (int step = 0; step < budget; step++) {
            AccessibilityNodeInfo root = root();
            Projection view = Projection.of(root);

            Projection.Item best = null;
            double bestScore = 0.0;
            for (Projection.Item item : view.items) {
                // Most things worth asserting about are also pressable, and plenty are not. When the
                // goal only looks, being pressable is neither required nor permission.
                if (!lookOnly && !item.tappable) {
                    continue;
                }
                double s = score(terms, item);
                if (s > bestScore) {
                    bestScore = s;
                    best = item;
                }
            }

            JSONObject record = new JSONObject();
            record.put("step", step);
            record.put("shown", view.items.size());
            record.put("more", view.more);

            if (best != null && bestScore >= SCORE_FLOOR) {
                if (lookOnly) {
                    record.put("n", "n" + best.index);
                    record.put("label", best.label());
                    record.put("score", round(bestScore));
                    record.put("decision", "done");
                    record.put("looked", true);
                    steps.put(record);
                    stop = "done";
                    break;
                }

                String key = progressKey(best);
                int already = tried.containsKey(key) ? tried.get(key) : 0;
                record.put("n", "n" + best.index);
                record.put("label", best.label());
                record.put("score", round(bestScore));
                if (already > 0) {
                    record.put("tried", already);
                    record.put("last", last.get(key));
                }

                // The goal's own target is the thing that stalled, so pressing it again is the
                // loop this closes. The screen has already settled by the time we get here, so a
                // second identical tap gets an identical non-result — and before this, that spent
                // the entire budget re-pressing one row, which read in a log as a helper that had
                // hung rather than a rule with nothing left to try.
                //
                // Only the *winner* is consulted, which is the distinction the corpus exists to
                // teach: a stalled node elsewhere on the screen means nothing, because the node
                // the goal is about has never been touched. Stalled nodes are therefore scored
                // normally and stay eligible to win; what changes is only what winning means when
                // the winner is the one that already failed.
                if (already > 0 && STALLED.contains(last.get(key))) {
                    record.put("decision", "handoff");
                    record.put("reason", "no_progress");
                    steps.put(record);
                    stop = "handoff";
                    break;
                }

                // Pressed already, and it worked. Measured live, this was the budget burn: one
                // control tapped three times in a row, every tap reporting {@code changed}, because
                // nothing had stalled and this loop had no notion of having finished.
                //
                // {@code done} claims the *action* is complete and nothing more: the control the
                // goal named was pressed and the screen moved. Whether the destination is right is
                // the caller's to judge — it holds the goal's success criteria and this does not.
                // That narrowness is why the check needs no vocabulary and no arrival heuristic.
                if (already > 0) {
                    record.put("decision", "done");
                    record.put("tried", already);
                    steps.put(record);
                    stop = "done";
                    break;
                }

                record.put("decision", "tap");
                String outcome = tap(best.node);
                boolean ok = !BLOCKED.equals(outcome);
                record.put("ok", ok);
                record.put("outcome", outcome);
                tried.put(key, already + 1);
                last.put(key, outcome);
                steps.put(record);
                if (!ok) {
                    stop = "tap_failed";
                    break;
                }
                // Arrival is the caller's to judge: it holds the goal's success criteria and this
                // does not. Reporting each step and stopping on budget keeps that boundary.
                continue;
            }

            boolean canScroll = view.more || hasScrollable(view);
            if (canScroll && scrolls < MAX_SCROLLS) {
                record.put("decision", "scroll");
                record.put("best_score", round(bestScore));
                boolean ok = scroll();
                record.put("ok", ok);
                steps.put(record);
                scrolls++;
                if (!ok) {
                    // Nothing moved, so there is no more screen to reveal and refusing is honest.
                    // Named the same way as the branch below, so every refusal on a consent screen
                    // reads `needs_auth` however the rule arrived at it.
                    stop = "handoff";
                    record.put("reason", atConsentGate(view) ? "needs_auth" : "target_absent");
                    break;
                }
                continue;
            }

            // Refusing. The only question left is what to call it.
            //
            // A consent dialog on screen makes `needs_auth` the better name, and this is the only
            // place the dialog is consulted — which is what makes the check safe. "A dialog is up"
            // must never be grounds to stop, because pressing "Don't allow" is a legitimate move.
            // Here nothing is being stopped: the rule had already given up. So whether the goal was
            // "about declining" never has to be decided from its text — if word overlap can reach
            // the decline control, the tap branch above already fired.
            //
            // `needs_auth` is also the more actionable name: it tells the host there is a consent
            // gate to answer, where `target_absent` tells it to give up.
            //
            // What is left is weak by construction: "nothing matched and nothing left to reveal"
            // cannot separate a target that is absent from one present under a name sharing no
            // *concept* with the goal. Spelling is handled; a synonym is not.
            record.put("decision", "handoff");
            record.put("reason", atConsentGate(view) ? "needs_auth" : "target_absent");
            record.put("best_score", round(bestScore));
            steps.put(record);
            stop = "handoff";
            break;
        }

        JSONObject out = new JSONObject();
        out.put("goal", goal);
        out.put("stop_reason", stop);
        out.put("steps", steps);
        out.put("step_count", steps.length());
        return out;
    }

    /**
     * The subject of a goal that asks what is on screen, or {@code null} if the goal asks for an
     * action. Mirrors {@code drive_rule.only_asks_to_look}.
     *
     * <p>The distinction is grammatical, not semantic, which is why it belongs in a rule: "is
     * doodads on screen" and "can you open doodads" name the same target and only one of them wants
     * it pressed. Returning the <em>subject</em> matters as much as the verdict, because the frame
     * words are goal text the scorer would otherwise rank nodes against, and real labels use them.
     */
    private static String onlyAsksToLook(String goal) {
        String text = goal == null ? "" : goal.trim().replaceAll("\\s+", " ");
        String lowered = text.toLowerCase(Locale.ROOT);

        for (String head : LOOK_HEADS) {
            if (lowered.startsWith(head)) {
                String subject = text.substring(head.length()).trim();
                return subject.isEmpty() ? null : subject;
            }
        }

        for (String tail : LOOK_TAILS) {
            if (!lowered.endsWith(tail)) {
                continue;
            }
            int cut = text.length() - tail.length();
            // "reach the solution screen" ends with the letters of "on screen" and is an
            // instruction. Five of this rule's six measured regressions were that, so the phrase
            // has to begin where a word begins.
            if (cut > 0 && text.charAt(cut - 1) != ' ') {
                continue;
            }
            String subject = text.substring(0, cut).trim();
            String stripped = subject.toLowerCase(Locale.ROOT);
            for (String lead : LOOK_LEADS) {
                if (stripped.startsWith(lead)) {
                    subject = subject.substring(lead.length()).trim();
                    break;
                }
            }
            return subject.isEmpty() ? null : subject;
        }

        return null;
    }

    /** Goal reduced to the words that could name a destination. */
    private static List<String> terms(String goal) {
        List<String> out = new ArrayList<>();
        for (String word : Projection.words(goal)) {
            if (word.length() > 1 && !STOPWORDS.contains(word)) {
                out.add(word);
            }
        }
        return out;
    }

    /** The first goal term naming a host capability, or {@code null} when none does. */
    private static String hostTerm(List<String> terms) {
        for (String term : terms) {
            if (HOST_TERMS.contains(term)) {
                return term;
            }
        }
        return null;
    }

    /** True when one of the listed nodes is a consent dialog's decline control. */
    private static boolean atConsentGate(Projection view) {
        for (Projection.Item item : view.items) {
            if (DECLINE_LABELS.contains(item.label().trim().toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    /**
     * The alphanumeric runs of a string with the original case kept — {@link Projection#words} with
     * the lowercasing switched off, because a capital is the separator this reads.
     */
    private static List<String> runs(String value) {
        List<String> out = new ArrayList<>();
        if (value == null || value.isEmpty()) {
            return out;
        }
        StringBuilder token = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
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

    /**
     * {@code AndroidWifi} split at its internal capitals; empty when there is no such boundary.
     *
     * <p>A boundary is a lowercase letter or digit followed by an uppercase one, which is how Android
     * names a compound label it has no room to space out.
     */
    private static List<String> caseParts(String token) {
        List<String> parts = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            char previous = current.length() > 0 ? current.charAt(current.length() - 1) : 0;
            if (Character.isUpperCase(c) && previous != 0
                    && (Character.isLowerCase(previous) || Character.isDigit(previous))) {
                parts.add(current.toString());
                current = new StringBuilder();
                current.append(c);
            } else {
                current.append(c);
            }
        }
        if (current.length() > 0) {
            parts.add(current.toString());
        }
        List<String> out = new ArrayList<>();
        if (parts.size() < 2) {
            return out;
        }
        for (String part : parts) {
            if (part.length() > 1) {
                out.add(part.toLowerCase(Locale.ROOT));
            }
        }
        return out;
    }

    /**
     * Every spelling of a label's words that a goal might use, {@link Projection#words} included.
     *
     * <p>Three forms, all mechanical, none a synonym: the words themselves; each word split at its
     * internal capitals ({@code AndroidWifi} gives {@code android}, {@code wifi}); and each adjacent
     * pair of short words joined ({@code Wi-Fi} gives {@code wifi}). Both directions are needed —
     * splitting reaches "Internet AndroidWifi", joining reaches "Reset Bluetooth &amp; Wi-Fi" — and
     * keeping the unsplit word is what stops {@code WiFi}, which already read correctly, from
     * breaking when the split turns it into {@code wi} + {@code fi}.
     *
     * <p>Deliberately <em>not</em> substring containment, which would reach the same two labels and
     * far more besides: 664 of the harvest's 2,993 label words are a proper substring of another, so
     * a goal about the {@code lock} screen would score against "Clock".
     */
    static List<String> variants(String label) {
        List<String> base = Projection.words(label);
        List<String> out = new ArrayList<>(base);
        for (String run : runs(label)) {
            out.addAll(caseParts(run));
        }
        for (int i = 0; i < base.size() - 1; i++) {
            String left = base.get(i);
            String right = base.get(i + 1);
            if (left.length() <= 6 && right.length() <= 6) {
                String joined = left + right;
                if (joined.length() >= 4 && joined.length() <= 14) {
                    out.add(joined);
                }
            }
        }
        return out;
    }

    /**
     * How well one node answers the goal.
     *
     * <p>Chrome scores zero so the ever-present search row cannot win by sharing a single word. A
     * hit on the node's first word counts for more than one later on, because an Android row reads
     * "Title Summary" and the head is where it names itself.
     */
    private static double score(List<String> terms, Projection.Item item) {
        String label = item.label();
        if (CHROME.contains(label.trim().toLowerCase(Locale.ROOT))) {
            return 0.0;
        }
        List<String> nodeWords = Projection.words(label);
        if (nodeWords.isEmpty() && item.rid.isEmpty()) {
            return 0.0;
        }
        // Membership is tested against every spelling; the head bonus is not. "First word" has to
        // keep meaning the row's actual first word, or a join two thirds down the summary would
        // collect the bonus meant for the row's own name.
        List<String> spellings = variants(label);
        List<String> ridWords = Projection.words(item.rid);
        double total = 0.0;
        for (int i = 0; i < terms.size(); i++) {
            String term = terms.get(i);
            if (spellings.contains(term)) {
                boolean head = !nodeWords.isEmpty() && nodeWords.get(0).equals(term);
                total += (head && i == 0) ? FIRST_TOKEN_BONUS : WORD_MATCH;
            } else if (ridWords.contains(term)) {
                total += RID_MATCH;
            }
        }
        return total;
    }

    private static boolean hasScrollable(Projection view) {
        for (Projection.Item item : view.items) {
            if (item.scrollable) {
                return true;
            }
        }
        return false;
    }

    /**
     * Click the node we chose, then wait for the screen to stop moving.
     *
     * <p>No selector is involved, so none can be ambiguous. The settle is not optional: without it
     * the loop re-read the tree before the transition finished, saw the screen it had just acted on,
     * chose the same row again, and acted on a node the framework had already recycled — which took
     * the accessibility service down with it and broke the channel for every later goal. Live, that
     * looked like "tap Display, tap Display, zero nodes, service gone".
     *
     * <p>{@link FlowFeature} already had this wait, correct, for every step it executes; reusing it
     * rather than writing a second one is the whole reason its helpers were made package-private.
     *
     * @return {@code changed}, {@code unchanged}, or {@code blocked} — the {@code last} value the
     *     next pass reads back. {@code unchanged} is the one that matters: it is a tap that landed
     *     and did nothing, and it is the only evidence {@code no_progress} ever gets.
     */
    private String tap(AccessibilityNodeInfo node) {
        AccessibilityNodeInfo target = FlowFeature.clickable(node);
        if (target == null) {
            return BLOCKED;
        }
        String before = flow.signature();
        if (!target.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
            return BLOCKED;
        }
        // `settle` already knows the answer — it is waiting on exactly this comparison — so it
        // reports it rather than being asked again from out here, which would mean a second
        // 400-node walk of the tree for something it has just computed.
        return flow.settle(before, FlowFeature.SETTLE_BUDGET_MS) ? CHANGED : UNCHANGED;
    }

    /**
     * What identifies a node across re-projections, for counting attempts against it.
     *
     * <p>Not the ordinal: {@code n3} is positional and every re-read of the tree renumbers, so
     * counting against it would credit one row's stall to whatever landed in its slot next. The
     * resource id plus the label is what stays put while the screen does, and the projection has
     * already collapsed duplicate labels onto their actionable twin, so a collision costs two rows
     * a shared counter and nothing worse.
     */
    private static String progressKey(Projection.Item item) {
        return item.rid + "\u0000" + item.label();
    }

    /** Delegated so movement detection and settle behaviour have exactly one implementation. */
    private boolean scroll() {
        try {
            JSONObject step = new JSONObject();
            step.put("kind", "scroll");
            step.put("arg", "up");
            JSONArray steps = new JSONArray();
            steps.put(step);
            JSONObject params = new JSONObject();
            params.put("steps", steps);
            JSONObject result = flow.handle("flow.run", params);
            return result.optInt("ok_count", result.optInt("total", 0)) > 0
                    || result.optBoolean("ok", false);
        } catch (Exception e) {
            // `flow.run` throws when a scroll moved nothing, which is the answer, not an error.
            return false;
        }
    }

    private AccessibilityNodeInfo root() {
        AccessibilityService service = HelperService.awaitService(2500L);
        if (service == null) {
            throw new IllegalStateException("accessibility service is not attached");
        }
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        if (root == null) {
            throw new IllegalStateException("no active window to read");
        }
        return root;
    }

    private static double round(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
