package dev.aua.helper;

import android.accessibilityservice.AccessibilityService;
import android.graphics.Rect;
import android.view.accessibility.AccessibilityNodeInfo;
import android.view.accessibility.AccessibilityWindowInfo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;
import java.util.function.Supplier;

/**
 * {@code ui.tree} — read the view hierarchy in-process.
 *
 * <p>Two things this buys over `adb shell uiautomator dump`: no adb round-trip for the
 * payload, and no UiAutomation slot held, so uiautomator2 and Maestro keep working
 * alongside the helper.
 *
 * <p>Field names mirror the attributes AUA's Android tree normalizer already reads, so the
 * host can map this onto the same Element schema as the XML path.
 */
final class TreeFeature implements Feature {

    /** Guards against a pathological tree turning one request into an OOM. */
    private static final int MAX_NODES = 4000;

    /**
     * Reads the *currently* bound service. The channel outlives any single service instance,
     * so holding one directly would leave this feature reading a dead instance after a rebind.
     */
    private final Supplier<AccessibilityService> current;

    TreeFeature(Supplier<AccessibilityService> current) {
        this.current = current;
    }

    @Override
    public String namespace() {
        return "ui";
    }

    @Override
    public JSONObject handle(String method, JSONObject params) throws Exception {
        if (!"ui.tree".equals(method)) {
            throw new IllegalArgumentException("unknown method: " + method);
        }
        boolean allWindows = params.optBoolean("all_windows", false);
        JSONObject out = new JSONObject();
        JSONArray roots = new JSONArray();
        int[] budget = new int[]{MAX_NODES};

        if (allWindows) {
            AccessibilityService service = requireService();
            List<AccessibilityWindowInfo> windows = service.getWindows();
            for (AccessibilityWindowInfo w : windows) {
                AccessibilityNodeInfo root = w.getRoot();
                if (root != null) {
                    JSONObject node = serialize(root, budget);
                    node.put("window_type", w.getType());
                    node.put("window_id", w.getId());
                    roots.put(node);
                }
            }
        } else {
            AccessibilityNodeInfo root = requireService().getRootInActiveWindow();
            if (root != null) {
                roots.put(serialize(root, budget));
            }
        }

        out.put("roots", roots);
        out.put("truncated", budget[0] <= 0);
        out.put("ts", System.currentTimeMillis());
        return out;
    }

    /**
     * Waits briefly for a bound service instead of failing the instant one is missing.
     *
     * <p>Anything that takes a UiAutomation connection — uiautomator2, which AUA itself uses —
     * makes the framework tear down and re-create accessibility services. That leaves a short
     * window with no bound instance, and a request landing in it used to fail outright even
     * though the service was healthy a moment either side.
     */
    private AccessibilityService requireService() {
        AccessibilityService s = HelperService.awaitService(2500L);
        if (s == null) {
            throw new IllegalStateException("accessibility service is not attached");
        }
        return s;
    }

    private JSONObject serialize(AccessibilityNodeInfo node, int[] budget) throws Exception {
        JSONObject o = new JSONObject();
        if (budget[0]-- <= 0) {
            return o;
        }
        Rect b = new Rect();
        node.getBoundsInScreen(b);

        o.put("class", str(node.getClassName()));
        o.put("package", str(node.getPackageName()));
        o.put("text", str(node.getText()));
        o.put("desc", str(node.getContentDescription()));
        o.put("rid", str(node.getViewIdResourceName()));
        // Same "[l,t][r,b]" shape the XML dump uses, so one parser serves both paths.
        o.put("bounds", "[" + b.left + "," + b.top + "][" + b.right + "," + b.bottom + "]");
        o.put("clickable", node.isClickable());
        o.put("long_clickable", node.isLongClickable());
        o.put("checkable", node.isCheckable());
        o.put("checked", node.isChecked());
        o.put("enabled", node.isEnabled());
        o.put("focused", node.isFocused());
        o.put("focusable", node.isFocusable());
        o.put("scrollable", node.isScrollable());
        o.put("selected", node.isSelected());
        o.put("password", node.isPassword());
        o.put("visible", node.isVisibleToUser());

        int count = node.getChildCount();
        if (count > 0) {
            JSONArray kids = new JSONArray();
            for (int i = 0; i < count; i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) {
                    kids.put(serialize(child, budget));
                }
            }
            o.put("children", kids);
        }
        return o;
    }

    private static String str(CharSequence cs) {
        return cs == null ? "" : cs.toString();
    }
}
