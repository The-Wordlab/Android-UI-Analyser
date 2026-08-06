package dev.androiduianalyser.companion;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.net.VpnService;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.provider.Settings;
import android.security.KeyChain;
import android.util.Base64;
import android.util.Log;

import java.io.OutputStream;
import java.io.ByteArrayInputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.Arrays;

import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;

public final class MainActivity extends Activity {
    private static final String LOG_TAG = "AUA_COMPANION";
    private static final int INSTALL_CA = 10;
    private static final int AUTHORIZE_VPN = 11;
    private Intent config;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        configure(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        configure(intent);
    }

    private void configure(Intent intent) {
        config = intent;
        String caBase64 = intent.getStringExtra("ca_base64");
        String caSha = intent.getStringExtra("ca_sha");
        String installedSha = getPreferences(MODE_PRIVATE).getString("ca_sha", "");
        if (caBase64 != null && caSha != null && !caSha.equals(installedSha)) {
            try {
                byte[] certificate = Base64.decode(caBase64, Base64.DEFAULT);
                Intent install;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    ContentValues values = new ContentValues();
                    values.put(MediaStore.MediaColumns.DISPLAY_NAME, "aua-interception-ca.crt");
                    values.put(MediaStore.MediaColumns.MIME_TYPE, "application/x-x509-ca-cert");
                    values.put(MediaStore.MediaColumns.RELATIVE_PATH, "Download");
                    values.put(MediaStore.MediaColumns.IS_PENDING, 1);
                    Uri uri = getContentResolver().insert(
                        MediaStore.Downloads.EXTERNAL_CONTENT_URI, values
                    );
                    if (uri == null) throw new IllegalStateException("Could not create CA file");
                    try (OutputStream output = getContentResolver().openOutputStream(uri)) {
                        if (output == null) throw new IllegalStateException("Could not open CA file");
                        output.write(certificate);
                    }
                    values.clear();
                    values.put(MediaStore.MediaColumns.IS_PENDING, 0);
                    getContentResolver().update(uri, values, null, null);
                    install = new Intent(Settings.ACTION_SECURITY_SETTINGS);
                } else {
                    install = KeyChain.createInstallIntent();
                    install.putExtra(KeyChain.EXTRA_NAME, "AUA interception CA");
                    install.putExtra(KeyChain.EXTRA_CERTIFICATE, certificate);
                }
                startActivityForResult(install, INSTALL_CA);
                return;
            } catch (Exception error) {
                Log.e(LOG_TAG, "Could not open CA installation", error);
                finish();
                return;
            }
        }
        authorizeVpn();
    }

    private void authorizeVpn() {
        Intent permission = VpnService.prepare(this);
        if (permission == null) {
            startTunnel();
        } else {
            startActivityForResult(permission, AUTHORIZE_VPN);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == INSTALL_CA) {
            byte[] certificate = Base64.decode(
                config.getStringExtra("ca_base64"), Base64.DEFAULT
            );
            boolean installed = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
                ? isCaTrusted(certificate)
                : resultCode == RESULT_OK;
            if (!installed) {
                Log.e(LOG_TAG, "CA installation was not approved");
                finish();
                return;
            }
            getPreferences(MODE_PRIVATE).edit()
                .putString("ca_sha", config.getStringExtra("ca_sha"))
                .apply();
            authorizeVpn();
        } else if (requestCode == AUTHORIZE_VPN && resultCode == RESULT_OK) {
            startTunnel();
        } else {
            finish();
        }
    }

    private void startTunnel() {
        Intent service = new Intent(this, AuaVpnService.class);
        service.putExtras(config);
        startForegroundService(service);
        finish();
    }

    private boolean isCaTrusted(byte[] encoded) {
        try {
            X509Certificate expected = (X509Certificate) CertificateFactory
                .getInstance("X.509")
                .generateCertificate(new ByteArrayInputStream(encoded));
            TrustManagerFactory factory = TrustManagerFactory.getInstance(
                TrustManagerFactory.getDefaultAlgorithm()
            );
            factory.init((java.security.KeyStore) null);
            for (TrustManager manager : factory.getTrustManagers()) {
                if (!(manager instanceof X509TrustManager)) continue;
                for (X509Certificate accepted : ((X509TrustManager) manager).getAcceptedIssuers()) {
                    if (Arrays.equals(expected.getEncoded(), accepted.getEncoded())) return true;
                }
            }
        } catch (Exception error) {
            Log.e(LOG_TAG, "Could not inspect trusted CAs", error);
        }
        return false;
    }
}
