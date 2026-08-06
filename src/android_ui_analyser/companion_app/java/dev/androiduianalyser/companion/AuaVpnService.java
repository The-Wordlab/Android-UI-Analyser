package dev.androiduianalyser.companion;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.net.ProxyInfo;
import android.net.VpnService;
import android.os.ParcelFileDescriptor;

import com.github.shadowsocks.bg.Tun2proxy;

import java.io.IOException;

public final class AuaVpnService extends VpnService {
    private static final int MTU = 1500;
    private static final String CHANNEL = "aua_companion_vpn";
    private ParcelFileDescriptor tunnel;
    private Thread worker;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        stopTunnel();
        createNotification();

        int port = intent.getIntExtra("proxy_port", 0);
        String targetPackage = intent.getStringExtra("target_package");
        if (port <= 0) {
            stopSelf();
            return START_NOT_STICKY;
        }

        Builder builder = new Builder()
            .setSession("AUA interception")
            .setMtu(MTU)
            .addAddress("10.73.0.2", 24)
            .addRoute("0.0.0.0", 0)
            .addDnsServer("8.8.8.8")
            .setHttpProxy(ProxyInfo.buildDirectProxy("127.0.0.1", port));
        try {
            if (targetPackage == null || targetPackage.trim().isEmpty()) {
                builder.addDisallowedApplication(getPackageName());
            } else {
                builder.addAllowedApplication(targetPackage);
            }
        } catch (Exception error) {
            stopSelf();
            return START_NOT_STICKY;
        }

        tunnel = builder.establish();
        if (tunnel == null) {
            stopSelf();
            return START_NOT_STICKY;
        }
        int fd = tunnel.getFd();
        String args = "tun2proxy --tun-fd " + fd
            + " --close-fd-on-drop false --proxy http://127.0.0.1:" + port
            + " --dns over-tcp --verbosity info";
        worker = new Thread(new Runnable() {
            @Override
            public void run() {
                Tun2proxy.run(args, (char) MTU);
            }
        }, "aua-tun2proxy");
        worker.start();
        return START_REDELIVER_INTENT;
    }

    private void createNotification() {
        NotificationManager notifications = getSystemService(NotificationManager.class);
        notifications.createNotificationChannel(
            new NotificationChannel(CHANNEL, "AUA interception", NotificationManager.IMPORTANCE_LOW)
        );
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(
            this, 0, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );
        Notification notification = new Notification.Builder(this, CHANNEL)
            .setContentTitle("AUA traffic interception active")
            .setContentText("Device traffic is routed to the local AUA proxy")
            .setSmallIcon(android.R.drawable.stat_sys_warning)
            .setContentIntent(pending)
            .setOngoing(true)
            .build();
        startForeground(1, notification);
    }

    private void stopTunnel() {
        Tun2proxy.stop();
        if (worker != null) worker.interrupt();
        if (tunnel != null) {
            try {
                tunnel.close();
            } catch (IOException ignored) {
            }
        }
        worker = null;
        tunnel = null;
    }

    @Override
    public void onDestroy() {
        stopTunnel();
        super.onDestroy();
    }
}
