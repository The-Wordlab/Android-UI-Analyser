# aua-fast

C thin client for the warm `aua` daemon. Skips Python/typer startup when the daemon
is already listening on its unix socket.

```bash
cd native/aua-fast && make && make install   # → ~/.local/bin/aua-fast
aua daemon start --quiet
aua-fast analyze          # ~ms instead of hundreds of ms cold Python
aua-fast tap 4
aua-fast has "Sign in"    # exit 0/1
```

If the daemon is down, `aua-fast` **exec's** `aua` on PATH (full CLI, all flags).

Socket path: `$AUA_DAEMON_SOCKET` or `~/.cache/android-ui-analyser/daemon.sock`.

Supported hot commands: `ping`, `analyze`, `devices`, `has`, `tap`, `key`, `input`,
`swipe`, `wait`. Everything else falls through to Python `aua`.
