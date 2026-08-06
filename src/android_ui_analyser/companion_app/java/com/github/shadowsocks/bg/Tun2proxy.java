package com.github.shadowsocks.bg;

public final class Tun2proxy {
    static {
        System.loadLibrary("tun2proxy");
    }

    private Tun2proxy() {
    }

    public static native int run(String args, char mtu);

    public static native int stop();
}
