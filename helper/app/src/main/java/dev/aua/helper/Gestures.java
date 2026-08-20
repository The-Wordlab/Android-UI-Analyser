package dev.aua.helper;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;
import android.graphics.Rect;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Synchronous gesture dispatch.
 *
 * <p>{@code dispatchGesture} is asynchronous and reports through a callback, but a flow step
 * is only finished when its gesture is. Every call here therefore blocks until the framework
 * says completed or cancelled — a step that returned early would let the next one act on a
 * screen still mid-swipe, which is the same fast-and-wrong failure that made an early
 * {@code flow.run} report success on a run that never navigated.
 *
 * <p>Path geometry mirrors the host's ``_swipe_path`` exactly: the same percent-of-box span
 * around the centre and the same one-pixel inset from the edges. The inset is not cosmetic —
 * a gesture starting on the very edge is claimed by the system's back and notification
 * gestures instead of by the list underneath.
 */
final class Gestures {

    /** Matches the host default for both swipe and scroll. */
    static final int DEFAULT_PERCENT = 70;

    private static final long TAP_MS = 60L;
    private static final long SWIPE_MS = 300L;
    /** A gesture cannot outlive this; the framework itself would be wedged. */
    private static final long DISPATCH_TIMEOUT_MS = 10_000L;

    private Gestures() {
    }

    static boolean tap(AccessibilityService service, int x, int y) {
        Path path = new Path();
        path.moveTo(x, y);
        return dispatch(service, path, TAP_MS);
    }

    /**
     * Swipe *direction* across *box*, using the host's geometry.
     *
     * @param box the area to gesture within — the whole screen for ``swipe``, the scrollable
     *            container for ``scroll``
     */
    static boolean swipe(AccessibilityService service, Rect box, String direction, int percent) {
        int cx = (box.left + box.right) / 2;
        int cy = (box.top + box.bottom) / 2;
        int spanX = Math.max(1, (box.right - box.left) * Math.min(percent, 90) / 200);
        int spanY = Math.max(1, (box.bottom - box.top) * Math.min(percent, 90) / 200);

        int x1;
        int y1;
        int x2;
        int y2;
        switch (direction.toLowerCase()) {
            case "up":
                x1 = cx; y1 = cy + spanY; x2 = cx; y2 = cy - spanY;
                break;
            case "down":
                x1 = cx; y1 = cy - spanY; x2 = cx; y2 = cy + spanY;
                break;
            case "left":
                x1 = cx + spanX; y1 = cy; x2 = cx - spanX; y2 = cy;
                break;
            case "right":
                x1 = cx - spanX; y1 = cy; x2 = cx + spanX; y2 = cy;
                break;
            default:
                throw new IllegalArgumentException("unknown direction '" + direction + "'");
        }

        Rect screen = screenBounds(service);
        Path path = new Path();
        path.moveTo(clamp(x1, 1, screen.right - 2), clamp(y1, 1, screen.bottom - 2));
        path.lineTo(clamp(x2, 1, screen.right - 2), clamp(y2, 1, screen.bottom - 2));
        return dispatch(service, path, SWIPE_MS);
    }

    static Rect screenBounds(AccessibilityService service) {
        android.util.DisplayMetrics m = service.getResources().getDisplayMetrics();
        return new Rect(0, 0, m.widthPixels, m.heightPixels);
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static boolean dispatch(AccessibilityService service, Path path, long durationMs) {
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(new GestureDescription.StrokeDescription(path, 0L, durationMs))
                .build();

        CountDownLatch done = new CountDownLatch(1);
        boolean[] completed = new boolean[]{false};
        boolean accepted = service.dispatchGesture(
                gesture,
                new AccessibilityService.GestureResultCallback() {
                    @Override
                    public void onCompleted(GestureDescription description) {
                        completed[0] = true;
                        done.countDown();
                    }

                    @Override
                    public void onCancelled(GestureDescription description) {
                        done.countDown();
                    }
                },
                null);
        if (!accepted) {
            return false;
        }
        try {
            if (!done.await(DISPATCH_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                return false;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
        return completed[0];
    }
}
