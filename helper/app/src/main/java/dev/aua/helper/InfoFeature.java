package dev.aua.helper;

import android.os.Build;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * {@code helper.info} — handshake.
 *
 * <p>The host calls this first and refuses to use any feature the APK does not advertise,
 * so a stale helper on a device fails loudly instead of silently missing methods.
 */
final class InfoFeature implements Feature {

    /** Bump on any wire-visible change. The host checks it against its own expectation. */
    static final int PROTOCOL = 1;

    private final HelperChannel channel;
    private final String versionName;

    InfoFeature(HelperChannel channel, String versionName) {
        this.channel = channel;
        this.versionName = versionName;
    }

    @Override
    public String namespace() {
        return "helper";
    }

    @Override
    public JSONObject handle(String method, JSONObject params) throws Exception {
        if (!"helper.info".equals(method)) {
            throw new IllegalArgumentException("unknown method: " + method);
        }
        JSONObject out = new JSONObject();
        out.put("protocol", PROTOCOL);
        out.put("version", versionName);
        out.put("sdk", Build.VERSION.SDK_INT);
        out.put("features", new JSONArray(channel.namespaces()));
        // The channel deliberately outlives any one service instance, so "the channel
        // answered" does not mean "the accessibility service is attached". Report that
        // separately: it is the only authoritative readiness signal, and the host cannot
        // derive it reliably from outside (dumpsys races with every UiAutomation connect).
        out.put("service_bound", HelperService.awaitService(2500L) != null);
        out.put("ts", System.currentTimeMillis());
        return out;
    }
}
