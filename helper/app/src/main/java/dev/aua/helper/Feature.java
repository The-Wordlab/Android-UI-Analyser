package dev.aua.helper;

import org.json.JSONObject;

/**
 * One named capability the helper exposes to the host.
 *
 * <p>This is the extension point. A feature owns a method namespace ({@code "ui"} owns
 * {@code "ui.tree"}) and answers requests on it. Adding a feature — an on-device model,
 * a full agent loop, gesture dispatch — means implementing this and registering it in
 * {@link HelperService}; the transport, framing and event plumbing stay untouched.
 */
public interface Feature {

    /** Namespace this feature owns, e.g. {@code "ui"} for {@code "ui.tree"}. */
    String namespace();

    /**
     * Handle one request.
     *
     * @param method full method name, e.g. {@code "ui.tree"}
     * @param params caller arguments; never null (empty object when absent)
     * @return the {@code result} payload
     * @throws Exception any failure; reported to the host as {@code ok:false}
     */
    JSONObject handle(String method, JSONObject params) throws Exception;
}
