package dev.aua.helper;

import android.view.accessibility.AccessibilityEvent;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * {@code a11y.*} — push screen-change events to the host.
 *
 * <p>The host subscribes to the event types it cares about, because the useful set differs
 * per caller: a settle wait wants content/window changes, flow recording wants clicks and
 * scrolls, toast capture wants notification state. Filtering on-device keeps the channel
 * quiet instead of shipping every event and discarding most of them on the host.
 *
 * <p>Default subscription is empty — the helper is silent until asked.
 */
final class EventsFeature implements Feature {

    private final HelperChannel channel;
    private final Set<Integer> subscribed = Collections.synchronizedSet(new HashSet<>());

    EventsFeature(HelperChannel channel) {
        this.channel = channel;
    }

    @Override
    public String namespace() {
        return "a11y";
    }

    @Override
    public JSONObject handle(String method, JSONObject params) throws Exception {
        JSONObject out = new JSONObject();
        switch (method) {
            case "a11y.subscribe": {
                JSONArray types = params.optJSONArray("types");
                subscribed.clear();
                if (types == null) {
                    // No explicit list means "everything" — one call to get a firehose while
                    // exploring, without needing to know the type constants up front.
                    subscribed.add(-1);
                } else {
                    for (int i = 0; i < types.length(); i++) {
                        subscribed.add(types.getInt(i));
                    }
                }
                out.put("subscribed", subscribed.size());
                return out;
            }
            case "a11y.unsubscribe": {
                subscribed.clear();
                out.put("subscribed", 0);
                return out;
            }
            default:
                throw new IllegalArgumentException("unknown method: " + method);
        }
    }

    /** Called on the service's event thread; must stay cheap. */
    void onEvent(AccessibilityEvent event) {
        if (subscribed.isEmpty()) {
            return;
        }
        int type = event.getEventType();
        if (!subscribed.contains(-1) && !subscribed.contains(type)) {
            return;
        }
        try {
            JSONObject e = new JSONObject();
            e.put("event", "a11y");
            e.put("ts", System.currentTimeMillis());
            e.put("type", type);
            e.put("type_name", AccessibilityEvent.eventTypeToString(type));
            e.put("package", event.getPackageName() == null ? "" : event.getPackageName().toString());
            e.put("class", event.getClassName() == null ? "" : event.getClassName().toString());
            // The text is the whole point for a toast or a notification: those never appear
            // in a hierarchy dump, because they are gone before anything can go and look.
            // Without it the host learns that *something* was announced and nothing more.
            StringBuilder text = new StringBuilder();
            for (CharSequence part : event.getText()) {
                if (part != null && part.length() > 0) {
                    if (text.length() > 0) {
                        text.append(' ');
                    }
                    text.append(part);
                }
            }
            e.put("text", text.toString());
            e.put("desc", event.getContentDescription() == null
                    ? "" : event.getContentDescription().toString());
            channel.broadcast(e);
        } catch (Exception e) {
            // An unserializable event must never take down the accessibility callback — but
            // it must not vanish either. Swallowing silently here is exactly what hid the
            // main-thread write failure that made every event disappear.
            android.util.Log.w("AuaHelper", "dropped an event: " + e);
        }
    }
}
