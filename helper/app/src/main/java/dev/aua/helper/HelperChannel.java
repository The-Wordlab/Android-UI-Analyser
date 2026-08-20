package dev.aua.helper;

import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Newline-delimited JSON over loopback TCP. The host reaches it with `adb forward`.
 *
 * <p>Two directions share one socket:
 * <ul>
 *   <li>host to device request: {@code {"id":1,"method":"ui.tree","params":{}}}
 *       answered by {@code {"id":1,"ok":true,"result":{...}}}</li>
 *   <li>device to host event, unsolicited: {@code {"event":"a11y","ts":...,...}}</li>
 * </ul>
 *
 * <p>Every write goes through a per-client queue drained by that client's own writer thread.
 * That is not tidiness: events originate on the accessibility callback, which runs on the
 * main thread, and Android throws {@code NetworkOnMainThreadException} for socket I/O there.
 * Writing inline lost every event silently — that exception is a RuntimeException, so an
 * {@code IOException} handler misses it and the callback's catch-all swallowed it. Replies
 * kept arriving because they are written on the reader thread, which made the failure look
 * like a broadcast bug rather than a threading one.
 *
 * <p>Loopback-only bind is deliberate: nothing off-device can reach this, so the helper
 * cannot become a remote control surface if it is left installed.
 */
public final class HelperChannel {

    private static final String TAG = "AuaHelper";

    /** Bounded so a stalled host cannot grow the queue without limit; oldest is dropped. */
    private static final int OUTBOX = 512;

    private final int port;
    private final Map<String, Feature> features = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<Client> clients = new CopyOnWriteArrayList<>();

    private volatile ServerSocket server;
    private volatile boolean running;

    public HelperChannel(int port) {
        this.port = port;
    }

    public void register(Feature feature) {
        features.put(feature.namespace(), feature);
    }

    /** Namespaces currently registered, for {@code helper.info}. */
    public String[] namespaces() {
        return features.keySet().toArray(new String[0]);
    }

    public void start() {
        if (running) {
            return;
        }
        running = true;
        new Thread(this::acceptLoop, "aua-helper-accept").start();
    }

    public void stop() {
        running = false;
        ServerSocket s = server;
        if (s != null) {
            try {
                s.close();
            } catch (IOException ignored) {
                // Closing the listener is what unblocks accept(); failure here is terminal anyway.
            }
        }
        for (Client c : clients) {
            c.close();
        }
        clients.clear();
    }

    private void acceptLoop() {
        try {
            // Backlog of 4: the host opens one connection, but a reconnect during teardown
            // should not be refused.
            server = new ServerSocket(port, 4, InetAddress.getByName("127.0.0.1"));
            Log.i(TAG, "channel listening on 127.0.0.1:" + port);
            while (running) {
                Socket socket = server.accept();
                try {
                    Client client = new Client(socket);
                    clients.add(client);
                    client.startThreads();
                } catch (IOException e) {
                    // One bad connection must not end the accept loop for every later one.
                    Log.e(TAG, "rejected a client: " + e);
                }
            }
        } catch (IOException e) {
            if (running) {
                Log.e(TAG, "accept loop ended: " + e);
            }
        }
    }

    /** Fan out an unsolicited event. Safe to call from the accessibility callback. */
    public void broadcast(JSONObject event) {
        String line = event.toString();
        for (Client c : clients) {
            c.enqueue(line);
        }
    }

    private JSONObject dispatch(String method, JSONObject params) throws Exception {
        int dot = method.indexOf('.');
        String ns = dot < 0 ? method : method.substring(0, dot);
        Feature feature = features.get(ns);
        if (feature == null) {
            throw new IllegalArgumentException("no feature for method: " + method);
        }
        return feature.handle(method, params == null ? new JSONObject() : params);
    }

    private final class Client {
        private final Socket socket;
        private final BufferedWriter out;
        private final BlockingQueue<String> outbox = new ArrayBlockingQueue<>(OUTBOX);
        private volatile boolean alive = true;

        Client(Socket socket) throws IOException {
            this.socket = socket;
            this.socket.setTcpNoDelay(true);
            this.out = new BufferedWriter(
                    new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        }

        void startThreads() {
            new Thread(this::readLoop, "aua-helper-rx").start();
            new Thread(this::writeLoop, "aua-helper-tx").start();
        }

        private void readLoop() {
            try (BufferedReader in = new BufferedReader(
                    new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while (alive && (line = in.readLine()) != null) {
                    handle(line);
                }
            } catch (IOException e) {
                Log.d(TAG, "client disconnected: " + e);
            } finally {
                close();
            }
        }

        private void writeLoop() {
            try {
                while (alive) {
                    String line = outbox.take();
                    out.write(line);
                    out.write('\n');
                    out.flush();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (IOException e) {
                Log.d(TAG, "client write ended: " + e);
            } finally {
                close();
            }
        }

        private void handle(String line) {
            if (line.trim().isEmpty()) {
                return;
            }
            JSONObject reply = new JSONObject();
            Object id = null;
            try {
                JSONObject req = new JSONObject(line);
                id = req.opt("id");
                JSONObject result = dispatch(req.getString("method"), req.optJSONObject("params"));
                reply.put("id", id).put("ok", true).put("result", result);
            } catch (Exception e) {
                try {
                    reply.put("id", id).put("ok", false)
                            .put("error", e.getClass().getSimpleName() + ": " + e.getMessage());
                } catch (Exception ignored) {
                    return;
                }
            }
            enqueue(reply.toString());
        }

        /** Never blocks and never throws: callable from the accessibility callback. */
        void enqueue(String line) {
            if (!alive) {
                return;
            }
            // A host that stops reading must not stall the device. Drop the oldest instead:
            // in a change-notification stream the newest state is the one that matters.
            while (!outbox.offer(line)) {
                if (outbox.poll() == null) {
                    return;
                }
            }
        }

        void close() {
            if (!alive) {
                return;
            }
            alive = false;
            clients.remove(this);
            try {
                socket.close();
            } catch (IOException ignored) {
                // Already gone; nothing useful to do.
            }
        }
    }
}
