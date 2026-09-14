package dev.aua.helper;

import android.accessibilityservice.AccessibilityService;
import android.os.Bundle;
import android.view.accessibility.AccessibilityNodeInfo;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * {@code model.run} — keep the complete observe/decide/act/verify loop in the helper process.
 *
 * <p>The host performs one bounded handoff and receives one result. Every intermediate hierarchy
 * read, model request, action, settle and contract check happens here. The OpenRouter credential is
 * deliberately a per-run value: it is removed from the request object immediately and is never
 * written to preferences, files, logs, result payloads or the APK. Compiling a bearer key into an
 * Android package would only turn the package into a convenient key extractor.
 *
 * <p>This remains inside the same safety boundary as the rest of AUA. The model can choose only the
 * small UI tool set below, element ids expire with every frame, and an achieved claim is accepted
 * only when the deterministic checks supplied by the caller pass on observations made here.
 */
final class ModelFeature implements Feature {

    static final String MODEL = "deepseek/deepseek-v4.1-flash";
    private static final String ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";
    private static final int MAX_RESPONSE_BYTES = 2_000_000;
    private static final int DEFAULT_MAX_STEPS = 16;
    private static final long DEFAULT_TIME_LIMIT_MS = 180_000L;
    // Killing uiautomator2 can return before Android has delivered binder death. On the
    // comparison emulator the helper briefly passed its channel preflight, detached, then
    // rebound 5.4s later. The model runner has no reason to fail inside that safe handoff
    // window, and it still remains bounded by the run's wall-clock limit.
    private static final long SERVICE_REBIND_LIMIT_MS = 10_000L;
    private static final double DEFAULT_COST_LIMIT_USD = 0.05;
    private static final String SYSTEM =
            "You drive one Android QA goal using only the tools provided. Use only node ids from "
            + "the newest observation; they expire after every tool call. Make exactly one tool "
            + "call per response. Do not sign in, pay, delete, share externally, or accept sensitive "
            + "consent unless the goal explicitly authorizes it. Inspect the live UI, make the "
            + "smallest safe action, and call finish only when the supplied deterministic checks "
            + "are satisfied. A finish claim cannot override a failed check.";

    private final FlowFeature flow;

    ModelFeature(FlowFeature flow) {
        this.flow = flow;
    }

    @Override
    public String namespace() {
        return "model";
    }

    @Override
    public JSONObject handle(String method, JSONObject params) throws Exception {
        if (!"model.run".equals(method)) {
            throw new IllegalArgumentException("unknown method: " + method);
        }
        String apiKey = params.optString("api_key", "").trim();
        params.remove("api_key");
        if (apiKey.isEmpty()) {
            throw new IllegalArgumentException("model.run needs an ephemeral `api_key`");
        }
        String goal = params.optString("goal", "").trim();
        if (goal.isEmpty()) {
            throw new IllegalArgumentException("model.run needs a `goal`");
        }
        String model = params.optString("model", MODEL);
        if (!MODEL.equals(model)) {
            throw new IllegalArgumentException("this runner is pinned to " + MODEL);
        }
        JSONArray definitions = params.optJSONArray("checks");
        if (definitions == null || definitions.length() == 0) {
            throw new IllegalArgumentException("model.run needs at least one deterministic check");
        }
        int maxSteps = clamp(params.optInt("max_steps", DEFAULT_MAX_STEPS), 1, 32);
        long timeLimitMs = clampLong(
                params.optLong("time_limit_ms", DEFAULT_TIME_LIMIT_MS), 10_000L, 600_000L);
        double costLimit = params.optDouble("cost_limit_usd", DEFAULT_COST_LIMIT_USD);
        if (!Double.isFinite(costLimit) || costLimit <= 0.0 || costLimit > 1.0) {
            throw new IllegalArgumentException("cost_limit_usd must be in (0, 1]");
        }

        long began = System.currentTimeMillis();
        Metrics metrics = new Metrics(costLimit);
        List<Check> checks = Check.parse(definitions);
        JSONArray actions = new JSONArray();
        JSONArray frames = new JSONArray();
        JSONObject result = new JSONObject()
                .put("ok", false)
                .put("ran_on", "device")
                .put("model", MODEL)
                .put("endpoint", "openrouter")
                .put("goal", goal);

        try {
            JSONArray messages = new JSONArray().put(
                    new JSONObject().put("role", "system").put("content", SYSTEM));
            Observation observation = observe(0, checks);
            frames.put(observation.summary);
            messages.put(new JSONObject().put("role", "user").put(
                    "content", new JSONObject()
                            .put("goal", goal)
                            .put("checks", publicChecks(checks, observation))
                            .put("observation", observation.modelView)
                            .toString()));

            String stopReason = "step_limit";
            String claim = null;
            for (int step = 0; step < maxSteps; step++) {
                if (System.currentTimeMillis() - began >= timeLimitMs) {
                    stopReason = "time_limit";
                    break;
                }
                if (metrics.costUsd >= costLimit) {
                    stopReason = "cost_limit";
                    break;
                }

                String frameSignature = observation.signature;
                long requestBegan = System.currentTimeMillis();
                JSONObject response = request(apiKey, messages);
                metrics.requestMs.put(System.currentTimeMillis() - requestBegan);
                metrics.consume(response);

                JSONObject choice = response.getJSONArray("choices").getJSONObject(0);
                JSONObject message = choice.getJSONObject("message");
                JSONObject assistant = assistantMessage(message);
                messages.put(assistant);
                JSONArray calls = message.optJSONArray("tool_calls");
                if (calls == null || calls.length() != 1) {
                    messages.put(new JSONObject().put("role", "user").put("content",
                            "The response must contain exactly one native tool call."));
                    continue;
                }

                JSONObject call = calls.getJSONObject(0);
                String callId = call.getString("id");
                JSONObject function = call.getJSONObject("function");
                String name = function.getString("name");
                JSONObject arguments = arguments(function.opt("arguments"));
                JSONObject toolResult = new JSONObject().put("ok", false);
                JSONObject action = new JSONObject()
                        .put("step", step)
                        .put("tool", name)
                        .put("arguments", arguments);

                if ("finish".equals(name)) {
                    Observation finalObservation = observe(step + 1, checks);
                    observation = finalObservation;
                    frames.put(finalObservation.summary);
                    JSONArray checkResults = checkResults(checks, finalObservation);
                    boolean verified = allPassed(checkResults);
                    String outcome = arguments.optString("outcome", "");
                    claim = arguments.optString("note", "");
                    action.put("ok", verified).put("outcome", outcome);
                    actions.put(action);
                    if (verified && ("achieved".equals(outcome)
                            || "already_satisfied".equals(outcome))) {
                        stopReason = "verified";
                        break;
                    }
                    if ("blocked".equals(outcome) || "not_achievable".equals(outcome)) {
                        stopReason = outcome;
                        break;
                    }
                    toolResult.put("error", "finish_rejected")
                            .put("checks", checkResults)
                            .put("observation", finalObservation.modelView);
                } else {
                    // A model response can take seconds. Refuse a node action when Android changed
                    // underneath it rather than applying an expired n-id to a different screen.
                    boolean nodeTool = "tap_and_observe".equals(name)
                            || "input_and_observe".equals(name);
                    String now = flow.signature();
                    long actionBegan = System.currentTimeMillis();
                    if (nodeTool && !now.equals(frameSignature)) {
                        toolResult.put("error", "frame_changed_while_model_was_deciding");
                        action.put("ok", false).put("error", "stale_frame");
                    } else {
                        JSONObject executed = execute(name, arguments, observation.projection);
                        toolResult = executed;
                        action.put("ok", executed.optBoolean("ok", false));
                        if (executed.has("error")) {
                            action.put("error", executed.optString("error"));
                        }
                    }
                    metrics.actionMs.put(System.currentTimeMillis() - actionBegan);
                    observation = observe(step + 1, checks);
                    frames.put(observation.summary);
                    toolResult.put("observation", observation.modelView)
                            .put("checks", publicChecks(checks, observation));
                    actions.put(action);
                }

                messages.put(new JSONObject()
                        .put("role", "tool")
                        .put("tool_call_id", callId)
                        .put("content", toolResult.toString()));
            }

            Observation finalObservation = observe(frames.length(), checks);
            JSONArray finalChecks = checkResults(checks, finalObservation);
            boolean verified = allPassed(finalChecks);
            result.put("ok", verified)
                    .put("verified", verified)
                    .put("stop_reason", stopReason)
                    .put("claim", claim == null ? JSONObject.NULL : claim)
                    .put("checks", finalChecks)
                    .put("actions", actions)
                    .put("frames", frames)
                    .put("final_observation", finalObservation.summary);
        } catch (Exception e) {
            result.put("ok", false)
                    .put("verified", false)
                    .put("stop_reason", "error")
                    .put("error", e.getClass().getSimpleName() + ": " + safeMessage(e));
        } finally {
            // Drop our only explicit reference. Java cannot guarantee immediate String erasure,
            // which is why this is acceptable only as an ephemeral experimental handoff and never
            // as a credential compiled into or persisted by the APK.
            apiKey = null;
            result.put("duration_ms", System.currentTimeMillis() - began)
                    .put("metrics", metrics.json());
        }
        return result;
    }

    private Observation observe(int step, List<Check> checks) throws JSONException {
        AccessibilityService service = service();
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        Projection projection = root == null ? Projection.of(null) : Projection.of(root);
        String signature = flow.signature();
        String packageName = root == null || root.getPackageName() == null
                ? "" : root.getPackageName().toString();
        JSONObject modelView = new JSONObject()
                .put("frame", step)
                .put("package", packageName)
                .put("truncated", projection.more);
        JSONArray nodes = new JSONArray();
        for (Projection.Item item : projection.items) {
            JSONObject node = new JSONObject().put("id", "n" + item.index);
            if (!item.text.isEmpty()) {
                node.put("text", item.text);
            }
            if (!item.desc.isEmpty()) {
                node.put("description", item.desc);
            }
            if (!item.rid.isEmpty()) {
                node.put("resource_id", item.rid);
            }
            node.put("tappable", item.tappable).put("scrollable", item.scrollable);
            nodes.put(node);
        }
        modelView.put("nodes", nodes);
        for (Check check : checks) {
            check.observe(root, packageName, step);
        }
        JSONObject summary = new JSONObject()
                .put("frame", step)
                .put("package", packageName)
                .put("signature", Integer.toHexString(signature.hashCode()))
                .put("nodes", nodes.length());
        return new Observation(projection, signature, modelView, summary);
    }

    private JSONObject execute(String name, JSONObject args, Projection projection)
            throws Exception {
        AccessibilityService service = service();
        switch (name) {
            case "tap_and_observe": {
                Projection.Item item = requireNode(projection, args.optString("node", ""));
                String before = flow.signature();
                boolean dispatched = FlowFeature.clickable(item.node)
                        .performAction(AccessibilityNodeInfo.ACTION_CLICK);
                flow.settle(before, FlowFeature.SETTLE_BUDGET_MS);
                return new JSONObject().put("ok", dispatched);
            }
            case "input_and_observe": {
                Projection.Item item = requireNode(projection, args.optString("node", ""));
                String text = args.optString("text", "");
                if (text.length() > 1000) {
                    return new JSONObject().put("ok", false).put("error", "input_too_long");
                }
                Bundle bundle = new Bundle();
                bundle.putCharSequence(
                        AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
                boolean dispatched = item.node.performAction(
                        AccessibilityNodeInfo.ACTION_SET_TEXT, bundle);
                return new JSONObject().put("ok", dispatched);
            }
            case "swipe_and_observe": {
                String direction = args.optString("direction", "").toLowerCase(Locale.ROOT);
                if (!(direction.equals("up") || direction.equals("down")
                        || direction.equals("left") || direction.equals("right"))) {
                    return new JSONObject().put("ok", false).put("error", "bad_direction");
                }
                String before = flow.signature();
                boolean dispatched = Gestures.swipe(service, Gestures.screenBounds(service),
                        direction, Gestures.DEFAULT_PERCENT);
                flow.settle(before, FlowFeature.SETTLE_BUDGET_MS);
                return new JSONObject().put("ok", dispatched);
            }
            case "key_and_observe": {
                if (!"back".equals(args.optString("key", "").toLowerCase(Locale.ROOT))) {
                    return new JSONObject().put("ok", false).put("error", "unsupported_key");
                }
                String before = flow.signature();
                boolean dispatched = service.performGlobalAction(
                        AccessibilityService.GLOBAL_ACTION_BACK);
                flow.settle(before, FlowFeature.SETTLE_BUDGET_MS);
                return new JSONObject().put("ok", dispatched);
            }
            case "wait_and_observe": {
                long ms = clampLong(args.optLong("ms", 750L), 100L, 5000L);
                Thread.sleep(ms);
                return new JSONObject().put("ok", true).put("waited_ms", ms);
            }
            default:
                return new JSONObject().put("ok", false).put("error", "unknown_tool");
        }
    }

    private static Projection.Item requireNode(Projection projection, String id) {
        if (!id.matches("n[1-9][0-9]*")) {
            throw new IllegalArgumentException("node must be a current n-id");
        }
        int wanted = Integer.parseInt(id.substring(1));
        for (Projection.Item item : projection.items) {
            if (item.index == wanted) {
                return item;
            }
        }
        throw new IllegalArgumentException("node " + id + " is not in the current frame");
    }

    private static JSONObject request(String apiKey, JSONArray messages) throws Exception {
        JSONObject payload = new JSONObject()
                .put("model", MODEL)
                .put("messages", messages)
                .put("tools", tools())
                .put("max_tokens", 4096)
                .put("reasoning", new JSONObject().put("effort", "low").put("exclude", false))
                .put("provider", new JSONObject()
                        .put("allow_fallbacks", true)
                        .put("sort", "throughput")
                        .put("data_collection", "deny")
                        .put("max_price", new JSONObject()
                                .put("prompt", 0.30).put("completion", 1.20)))
                .put("plugins", new JSONArray().put(new JSONObject()
                        .put("id", "context-compression").put("enabled", false)))
                .put("usage", new JSONObject().put("include", true));

        HttpURLConnection connection = (HttpURLConnection) new URL(ENDPOINT).openConnection();
        connection.setRequestMethod("POST");
        connection.setConnectTimeout(15_000);
        connection.setReadTimeout(120_000);
        connection.setDoOutput(true);
        connection.setRequestProperty("Authorization", "Bearer " + apiKey);
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setRequestProperty("Accept", "application/json");
        connection.setRequestProperty("User-Agent", "AUA-Helper/0.1");
        byte[] body = payload.toString().getBytes(StandardCharsets.UTF_8);
        connection.setFixedLengthStreamingMode(body.length);
        try (OutputStream out = connection.getOutputStream()) {
            out.write(body);
        }
        int status = connection.getResponseCode();
        InputStream stream = status >= 200 && status < 300
                ? connection.getInputStream() : connection.getErrorStream();
        String response = readBounded(stream, MAX_RESPONSE_BYTES);
        connection.disconnect();
        if (status < 200 || status >= 300) {
            String detail = response.replace('\n', ' ').replace('\r', ' ');
            if (detail.length() > 500) {
                detail = detail.substring(0, 500);
            }
            throw new IOException("OpenRouter HTTP " + status + ": " + detail);
        }
        return new JSONObject(response);
    }

    private static String readBounded(InputStream stream, int limit) throws IOException {
        if (stream == null) {
            return "";
        }
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            char[] chunk = new char[8192];
            int read;
            while ((read = reader.read(chunk)) >= 0) {
                if (out.length() + read > limit) {
                    throw new IOException("model response exceeded " + limit + " bytes");
                }
                out.append(chunk, 0, read);
            }
        }
        return out.toString();
    }

    private static JSONObject assistantMessage(JSONObject message) throws JSONException {
        JSONObject out = new JSONObject().put("role", "assistant");
        String[] fields = {
            "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details"
        };
        for (String field : fields) {
            if (message.has(field)) {
                out.put(field, message.get(field));
            }
        }
        return out;
    }

    private static JSONObject arguments(Object raw) throws JSONException {
        if (raw instanceof JSONObject) {
            return (JSONObject) raw;
        }
        if (raw instanceof String && !((String) raw).trim().isEmpty()) {
            return new JSONObject((String) raw);
        }
        return new JSONObject();
    }

    private static JSONArray tools() throws JSONException {
        JSONArray tools = new JSONArray();
        tools.put(tool("tap_and_observe", "Tap one node from the current frame.",
                object(new JSONObject().put("node", string()), "node")));
        tools.put(tool("input_and_observe", "Replace the text in one current input node.",
                object(new JSONObject().put("node", string()).put("text", string()),
                        "node", "text")));
        tools.put(tool("swipe_and_observe", "Swipe the screen and observe the result.",
                object(new JSONObject().put("direction", new JSONObject()
                        .put("type", "string")
                        .put("enum", new JSONArray().put("up").put("down")
                                .put("left").put("right"))), "direction")));
        tools.put(tool("key_and_observe", "Press Android Back and observe the result.",
                object(new JSONObject().put("key", new JSONObject()
                        .put("type", "string")
                        .put("enum", new JSONArray().put("back"))), "key")));
        tools.put(tool("wait_and_observe", "Wait briefly for asynchronous UI work.",
                object(new JSONObject().put("ms", new JSONObject()
                        .put("type", "integer").put("minimum", 100).put("maximum", 5000)),
                        "ms")));
        tools.put(tool("finish", "Finish only after the deterministic checks pass.",
                object(new JSONObject()
                                .put("outcome", new JSONObject().put("type", "string")
                                        .put("enum", new JSONArray().put("achieved")
                                                .put("already_satisfied").put("blocked")
                                                .put("not_achievable")))
                                .put("note", string()),
                        "outcome", "note")));
        return tools;
    }

    private static JSONObject tool(String name, String description, JSONObject parameters)
            throws JSONException {
        return new JSONObject().put("type", "function").put("function", new JSONObject()
                .put("name", name).put("description", description).put("parameters", parameters));
    }

    private static JSONObject object(JSONObject properties, String... required)
            throws JSONException {
        JSONArray req = new JSONArray();
        for (String name : required) {
            req.put(name);
        }
        return new JSONObject().put("type", "object").put("properties", properties)
                .put("required", req).put("additionalProperties", false);
    }

    private static JSONObject string() throws JSONException {
        return new JSONObject().put("type", "string");
    }

    private AccessibilityService service() {
        AccessibilityService service = HelperService.awaitService(SERVICE_REBIND_LIMIT_MS);
        if (service == null) {
            throw new IllegalStateException("accessibility service is not attached");
        }
        return service;
    }

    private static JSONArray publicChecks(List<Check> checks, Observation observation)
            throws JSONException {
        JSONArray out = new JSONArray();
        for (Check check : checks) {
            out.put(check.publicState(observation));
        }
        return out;
    }

    private static JSONArray checkResults(List<Check> checks, Observation observation)
            throws JSONException {
        JSONArray out = new JSONArray();
        for (Check check : checks) {
            out.put(check.result(observation));
        }
        return out;
    }

    private static boolean allPassed(JSONArray checks) {
        for (int i = 0; i < checks.length(); i++) {
            if (!checks.optJSONObject(i).optBoolean("passed", false)) {
                return false;
            }
        }
        return true;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static long clampLong(long value, long min, long max) {
        return Math.max(min, Math.min(max, value));
    }

    private static String safeMessage(Exception error) {
        String message = error.getMessage();
        if (message == null) {
            return "no detail";
        }
        return message.length() > 600 ? message.substring(0, 600) : message;
    }

    private static final class Observation {
        final Projection projection;
        final String signature;
        final JSONObject modelView;
        final JSONObject summary;

        Observation(Projection projection, String signature, JSONObject modelView,
                JSONObject summary) {
            this.projection = projection;
            this.signature = signature;
            this.modelView = modelView;
            this.summary = summary;
        }
    }

    private static final class Metrics {
        final double limitUsd;
        final JSONArray requestMs = new JSONArray();
        final JSONArray actionMs = new JSONArray();
        int requests;
        int promptTokens;
        int completionTokens;
        int reasoningTokens;
        double costUsd;
        String provider = "";

        Metrics(double limitUsd) {
            this.limitUsd = limitUsd;
        }

        void consume(JSONObject response) throws JSONException {
            JSONObject usage = response.optJSONObject("usage");
            if (usage == null || !usage.has("cost")) {
                throw new JSONException("OpenRouter response omitted usage.cost; spend is unknown");
            }
            double cost = usage.optDouble("cost", Double.NaN);
            if (!Double.isFinite(cost) || cost < 0.0) {
                throw new JSONException("OpenRouter returned an invalid usage.cost");
            }
            requests++;
            costUsd += cost;
            promptTokens += usage.optInt("prompt_tokens", 0);
            completionTokens += usage.optInt("completion_tokens", 0);
            JSONObject details = usage.optJSONObject("completion_tokens_details");
            if (details != null) {
                reasoningTokens += details.optInt("reasoning_tokens", 0);
            }
            provider = response.optString("provider", provider);
        }

        JSONObject json() throws JSONException {
            return new JSONObject()
                    .put("requests", requests)
                    .put("prompt_tokens", promptTokens)
                    .put("completion_tokens", completionTokens)
                    .put("reasoning_tokens", reasoningTokens)
                    .put("reported_usd", costUsd)
                    .put("cost_limit_usd", limitUsd)
                    .put("provider", provider.isEmpty() ? JSONObject.NULL : provider)
                    .put("model_request_ms", requestMs)
                    .put("action_ms", actionMs)
                    .put("cost_boundary", "reported completed responses; one in-flight request may exceed limit");
        }
    }

    private static final class Check {
        final String id;
        final String kind;
        final String selector;
        final String value;
        boolean everMatched;
        boolean firstMatched;
        boolean violated;
        int evidenceFrame = -1;
        int observations;

        Check(String id, String kind, String selector, String value) {
            this.id = id;
            this.kind = kind;
            this.selector = selector;
            this.value = value;
        }

        static List<Check> parse(JSONArray raw) throws JSONException {
            List<Check> out = new ArrayList<>();
            for (int i = 0; i < raw.length(); i++) {
                JSONObject item = raw.getJSONObject(i);
                String id = item.optString("id", "").trim();
                String kind = item.optString("kind", "").trim();
                String selector = item.optString("selector", "").trim();
                String value = item.optString("value", "").trim();
                if (id.isEmpty() || value.isEmpty()) {
                    throw new JSONException("every check needs non-empty id and value");
                }
                if (!(kind.equals("first_visible") || kind.equals("ever_visible")
                        || kind.equals("final_visible") || kind.equals("final_absent")
                        || kind.equals("never_visible"))) {
                    throw new JSONException("unsupported check kind " + kind);
                }
                if (!(selector.equals("rid") || selector.equals("text")
                        || selector.equals("desc") || selector.equals("package"))) {
                    throw new JSONException("unsupported check selector " + selector);
                }
                out.add(new Check(id, kind, selector, value));
            }
            return out;
        }

        void observe(AccessibilityNodeInfo root, String packageName, int frame) {
            boolean matched = matches(root, packageName);
            if (observations == 0) {
                firstMatched = matched;
            }
            observations++;
            if (matched) {
                everMatched = true;
                if (evidenceFrame < 0) {
                    evidenceFrame = frame;
                }
                if (kind.equals("never_visible")) {
                    violated = true;
                }
            }
        }

        JSONObject publicState(Observation observation) throws JSONException {
            return new JSONObject()
                    .put("id", id)
                    .put("kind", kind)
                    .put("selector", selector)
                    .put("value", value)
                    .put("currently_matches", matchesCurrent(observation));
        }

        JSONObject result(Observation observation) throws JSONException {
            boolean current = matchesCurrent(observation);
            boolean passed;
            switch (kind) {
                case "first_visible":
                    passed = firstMatched;
                    break;
                case "ever_visible":
                    passed = everMatched;
                    break;
                case "final_visible":
                    passed = current;
                    break;
                case "final_absent":
                    passed = !current;
                    break;
                case "never_visible":
                    passed = !violated;
                    break;
                default:
                    passed = false;
            }
            return new JSONObject()
                    .put("id", id)
                    .put("kind", kind)
                    .put("passed", passed)
                    .put("evidence_frame", evidenceFrame < 0 ? JSONObject.NULL : evidenceFrame);
        }

        private boolean matchesCurrent(Observation observation) {
            // Package checks are exact enough in the compact summary. UI selectors were evaluated
            // on the full live hierarchy in observe(), so the current frame is the latest result.
            if (selector.equals("package")) {
                return observation.modelView.optString("package", "")
                        .equalsIgnoreCase(value);
            }
            // For final selectors re-read the live tree. Projection intentionally omits many rids.
            AccessibilityService service = HelperService.awaitService(2500L);
            AccessibilityNodeInfo root = service == null ? null : service.getRootInActiveWindow();
            return matches(root, observation.modelView.optString("package", ""));
        }

        private boolean matches(AccessibilityNodeInfo root, String packageName) {
            if (selector.equals("package")) {
                return packageName.equalsIgnoreCase(value);
            }
            return matchesNode(root);
        }

        private boolean matchesNode(AccessibilityNodeInfo node) {
            if (node == null) {
                return false;
            }
            String candidate;
            switch (selector) {
                case "rid":
                    candidate = normal(node.getViewIdResourceName());
                    String wanted = normal(value);
                    if (candidate.equals(wanted) || candidate.endsWith("/" + wanted)) {
                        return true;
                    }
                    break;
                case "text":
                    candidate = normal(node.getText());
                    if (candidate.contains(normal(value))) {
                        return true;
                    }
                    break;
                case "desc":
                    candidate = normal(node.getContentDescription());
                    if (candidate.contains(normal(value))) {
                        return true;
                    }
                    break;
                default:
                    break;
            }
            for (int i = 0; i < node.getChildCount(); i++) {
                if (matchesNode(node.getChild(i))) {
                    return true;
                }
            }
            return false;
        }

        private static String normal(CharSequence value) {
            return value == null ? "" : value.toString().trim().toLowerCase(Locale.ROOT);
        }
    }
}
