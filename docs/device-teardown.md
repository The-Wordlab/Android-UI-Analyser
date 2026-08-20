# Device teardown — undoing changes the agent walked away from

## The problem

`aua proxy start` sets `settings put global http_proxy 127.0.0.1:<port>` and opens an
`adb reverse` tunnel to a host `mitmdump`. Three lifetimes, none of them the same:

| thing | lives until |
|---|---|
| `http_proxy` (device-global setting) | someone changes it, or the device reboots |
| `adb reverse` tunnel | the adb transport drops |
| `mitmdump` | it is killed — it is spawned with `start_new_session=True`, so it survives its parent |

Kill the agent and the device keeps pointing at a port nothing serves. Every app reports
"Offline", `NetworkMonitor` marks the network unvalidated (the Wi-Fi "!"), and the next agent
inherits a device that looks broken for reasons that have nothing to do with the app under test.

Measured on a developer host, 2026-08-19: two orphan `mitmdump` processes two days old, still
holding their listen ports; zero proxy ownership records; three emulators leased by processes that
no longer existed. The same shape applies to verified-offline mode, a moved wall clock, an applied
radio profile, disabled animations, and the on-device helper's accessibility service.

## Why nothing caught it

`leases.py` is deliberately reaper-free: expiry is computed when a lease is **read**, and a lease
whose owner pid is gone reads as expired immediately, before its TTL is consulted. That makes a
crashed agent's lease self-healing and a permanent block impossible — a good property — but it also
means **nothing executes at the moment a lease ends**. There is no hook to hang cleanup on.

`proxy_mock.py` already had the right idea (`write_state` / `read_state` / `orphan_reason`, keyed by
serial at a fixed cross-process path) with the right comment explaining why. It had zero callers.

## The design

### 1. A write-ahead undo ledger

`device_ledger.py` journals one entry per reversible change, at
`~/.cache/android-ui-analyser/device-state/<serial>.json`.

```json
{
  "key": "http_proxy",
  "kind": "http_proxy",
  "op": "set_http_proxy",
  "args": {"host_port": null},
  "owner": "claude-72860-3:252026",
  "owner_pid": 72860,
  "owner_started": "3:252026",
  "instance_token": "40b8403f-…",
  "cache_dir": "~/.cache/android-ui-analyser",
  "leased": true,
  "recorded": 1787166215.67
}
```

Four properties do the work:

- **Written before the device is touched.** A crash after the record leaves a redundant undo,
  which is harmless and idempotent. A crash before it leaves a device nobody can clean, and no
  watchdog — however alive — can fix what nothing wrote down.
- **At a fixed path, not under `cache.dir`.** Parallel agents are told to keep separate caches so
  their mock rules cannot leak into each other; that also means the port agent A wrote into its own
  cache is invisible to agent B, and B is the one that inherits the emulator.
- **The undo is data, not a closure.** `{"op": …, "args": …}` replayed through the selected
  `PlatformAdapter`, so an unrelated process — on any platform — can execute it.
- **Keyed, so re-recording replaces.** `reverse_port:49097` keeps two different tunnels apart while
  making the same change recorded twice a single undo.

`instance_token` is the device's boot id. A reboot already undid the change, so the entry is dropped
without touching the target — the same guard `network_profiles` applies to its own restore points.

### 2. Deciding when it is safe to replay

`device_ledger.reapable()` returns a reason, or `None` for "leave it alone". Undoing a running
agent's proxy mid-flow is strictly worse than leaving a stale one behind: the first breaks a test
that was working, the second is visible in `aua teardown status`. So every reason is positive
evidence that nobody is using the change:

1. A live lease for that serial in **any** cache dir the entries name → hands off.
2. Every recording process provably gone → reap now.
3. Leasing was on and the lease is gone → reap after `teardown.grace_s`. A long-lived orchestrator
   or `claude` process outliving the work is the *normal* case; waiting for it to exit would keep
   the first emulator dirty for hours.
4. Leasing was off (no ownership signal at all) → only a dead owner licenses a reap.

`grace_s` is deliberately **not** derived from `lease.ttl_s`. That 900s exists so a legitimate
90–120s `--until` wait cannot lose its device; this one bounds how long a device may stay dirty.
Two different questions, two numbers.

### 3. Two nets that replay it

**An opportunistic sweep**, once per Engine, when the device is first connected (`teardown.sweep`).
A directory glob in the normal case. Covers the common path: an agent walks away, the next agent's
first command cleans up before it starts. But it only fires if someone runs `aua` again — and the
last agent of the day is exactly the one that leaves a device dirty overnight.

**A detached per-device watchdog** (`teardown_watchdog.py`), spawned by the command that records the
first change — not by device boot, because a physical phone whose clock was moved has no emulator
process to hang a watchdog off, and that is the target where a leftover change hurts most. It
polls, replays when the holder is provably gone, and exits when the ledger is empty. A thread could
not do this job: the failure being covered is *that process dying*.

Plus two explicit hooks: `aua lease release` resets before handing the device back, and
`aua teardown run` forces it by hand.

### 4. The idle emulator watchdog, lease-gated

The existing idle watchdog stopped an aua-started headless emulator after wall-clock idleness alone.
That is a weak signal — an agent can legitimately sit in a 120s wait — so the timeout had to stay
long (900s) to be safe. It now also requires **no live lease**, which is a strong signal precisely
because a dead owner's lease expires instantly. That is what makes
`teardown.emulator_idle_stop_s: 120` a reasonable thing to configure; the default stays 900 so
existing users see no change. It reaps the device's ledger before stopping it, because the host-side
residue outlives the emulator.

An emulator AUA did not start is never touched.

## Adding a mutation

1. Add to `device_ledger.MUTATION_CATALOGUE`: the kind, the `module.py:function` that performs it,
   and the undo op (or `None` with a reason it needs none).
2. Add the op to `device_ledger.UNDO_OPS` if new — a handler taking `(UndoContext, args)`,
   reaching the device through `ctx.require_device()` / `ctx.require_capability(name)` only.
3. Call `Engine.record_device_change(...)` **before** the mutation, and
   `Engine.forget_device_change(...)` wherever you undo it deliberately.

`tests/test_every_device_mutation_registers_an_undo.py` scans the Android backends for mutating
command literals and fails, naming your call site, until it is attributed. It also refuses a
catalogue entry naming a handler that no longer exists, and an undo op no mutation records.

## Commands

```bash
aua teardown status                 # what is pending, per device, and whether it can run now
aua teardown run                    # undo everything no live agent still holds
aua teardown run --dry-run          # say what would happen, touch nothing
aua teardown run --serial-target emulator-5554 --force
```

## Configuration

```yaml
teardown:
  enabled: true
  sweep_on_command: true      # cheap net: glob the ledger when a device is connected
  watchdog: true              # detached net: survives the agent, the daemon, the shell
  grace_s: 120.0              # how long an unheld change is left alone
  watchdog_poll_s: 15.0
  emulator_idle_stop_s: 900.0 # stop an aua-started emulator idle AND unleased this long (0=off)
```
