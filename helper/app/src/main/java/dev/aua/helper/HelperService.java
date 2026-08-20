package dev.aua.helper;

import android.accessibilityservice.AccessibilityService;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import java.util.concurrent.atomic.AtomicReference;

/**
 * The accessibility service that hosts the helper channel.
 *
 * <p>An AccessibilityService is the only way to get pushed screen-change notifications and
 * in-process hierarchy reads. It must be switched on explicitly — the host does that over
 * adb on a rooted development device, and refuses to try anywhere else.
 *
 * <p>Everything useful lives in {@link Feature}s. This class only owns the accessibility
 * lifecycle and wires the registry, so a later feature (an on-device model, a full agent
 * loop, gesture dispatch) is an added registration rather than a change here.
 *
 * <p>The channel is process-wide and created once. Android rebinds an accessibility service
 * more than once in a process, and building a channel per bind produced a silent failure:
 * the second channel could not take the port, events were broadcast to whichever instance
 * was newest, and a host connected to the older socket received nothing while every request
 * still answered normally. The service instance is swapped behind an
 * {@link AtomicReference} instead, so features always read through the live one.
 */
public final class HelperService extends AccessibilityService {

    private static final String TAG = "AuaHelper";

    /** Loopback port on the device; the host maps it with `adb forward`. */
    public static final int PORT = 8779;

    /** The currently bound service, or null between binds. */
    static final AtomicReference<AccessibilityService> CURRENT = new AtomicReference<>();

    private static volatile HelperChannel channel;
    private static volatile EventsFeature events;
    private static volatile RecordFeature recorder;

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        CURRENT.set(this);
        // Every bind, not just the first. The process is cached and then frozen whenever this
        // service is torn down, which happens on every UiAutomation connection the host makes,
        // so the keep-alive has to be re-established each time the service comes back.
        KeepAliveService.start(this);
        synchronized (HelperService.class) {
            if (channel == null) {
                HelperChannel c = new HelperChannel(PORT);
                EventsFeature e = new EventsFeature(c);
                c.register(e);
                c.register(new TreeFeature(CURRENT::get));
                c.register(new FlowFeature());
                RecordFeature rec = new RecordFeature();
                c.register(rec);
                recorder = rec;
                c.register(new InfoFeature(c, BuildConfig.VERSION_NAME));
                c.start();
                channel = c;
                events = e;
                Log.i(TAG, "helper channel started on " + PORT);
            } else {
                Log.i(TAG, "helper rebound; reusing existing channel on " + PORT);
            }
        }
    }

    /**
     * The bound service, waiting up to {@code timeoutMs} for one to attach.
     *
     * <p>Anything that takes a UiAutomation connection — uiautomator2, which AUA itself uses
     * on every command — makes the framework detach and re-create accessibility services.
     * A caller checking at an arbitrary instant therefore sees "not attached" on a perfectly
     * healthy helper, so readiness is defined as "attaches within a short grace period".
     */
    static AccessibilityService awaitService(long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (true) {
            AccessibilityService s = CURRENT.get();
            if (s != null) {
                return s;
            }
            if (System.currentTimeMillis() >= deadline) {
                return null;
            }
            try {
                Thread.sleep(50L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        EventsFeature e = events;
        if (e != null) {
            e.onEvent(event);
        }
        RecordFeature r = recorder;
        if (r != null) {
            r.onEvent(event);
        }
    }

    @Override
    public void onInterrupt() {
        // Required by the framework. The channel is unaffected by an interrupt request.
    }

    @Override
    public boolean onUnbind(android.content.Intent intent) {
        // Deliberately does not stop the channel: a rebind is routine and tearing the socket
        // down here is what made reconnects race. The process only exists while this service
        // is enabled, so the channel dies with it.
        CURRENT.compareAndSet(this, null);
        return super.onUnbind(intent);
    }
}
