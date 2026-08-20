package dev.aua.helper;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;

/**
 * A foreground service whose only job is to stop Android freezing this process.
 *
 * <p>Nothing here talks to the host or the screen. It exists because of one line in logcat:
 *
 * <pre>D/ActivityManager: freezing 30850 dev.aua.helper</pre>
 *
 * <p>Android freezes cached processes, and this process is cached most of the time by design.
 * Whenever anything takes a UiAutomation connection — uiautomator2, which AUA itself uses on
 * nearly every command — the framework tears down every accessibility service, and a process
 * with no bound component is cached within seconds and then frozen. Handing work back to the
 * helper afterwards then failed in the least readable way possible: the framework reported the
 * service as bound, the helper answered its handshake, and a few steps into the run the calls
 * started throwing "accessibility service is not attached". Measured on a 24-step run, the
 * device completed between 3 and 11 steps and a different number every time.
 *
 * <p>A foreground service makes the process ineligible for caching, so it is never frozen and
 * the accessibility service rebinds immediately instead of after a thaw. The notification is
 * the price Android charges for that, and it is deliberately minimal: this is a development
 * tool that the user installed on purpose on a rooted device.
 */
public final class KeepAliveService extends Service {

    private static final String TAG = "AuaHelper";
    private static final String CHANNEL_ID = "aua-helper-keepalive";
    private static final int NOTIFICATION_ID = 1;

    /** Start (or no-op if already running). Safe to call on every accessibility rebind. */
    static void start(Context context) {
        Intent intent = new Intent(context, KeepAliveService.class);
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent);
            } else {
                context.startService(intent);
            }
        } catch (Exception e) {
            // Losing the keep-alive costs reliability, never correctness: the host's readiness
            // probe declines the handover rather than trusting a helper that cannot answer.
            Log.w(TAG, "keep-alive service refused to start: " + e);
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "AUA Helper", NotificationManager.IMPORTANCE_MIN);
            channel.setShowBadge(false);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
        Notification notification = buildNotification();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                    NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        Log.i(TAG, "keep-alive foreground service started");
    }

    private Notification buildNotification() {
        Notification.Builder builder =
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                        ? new Notification.Builder(this, CHANNEL_ID)
                        : new Notification.Builder(this);
        return builder
                .setContentTitle("AUA Helper")
                .setContentText("Ready for on-device UI automation")
                .setSmallIcon(android.R.drawable.stat_notify_sync_noanim)
                .setOngoing(true)
                .build();
    }

    /** Restart if Android kills it: being gone is exactly the state this service prevents. */
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
