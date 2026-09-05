# Device teardown — undoing changes the agent walked away from

## The problem

`aua proxy start` sets `settings put global http_proxy 127.0.0.1:<port>` and opens an
`adb reverse` tunnel to a host `mitmdump`. Three lifetimes, none of them the same:

| thing | lives until |
|---|---|
| `http_proxy` (device-global setting) | someone changes or resets the persisted setting |
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

`instance_token` identifies the target boot. A changed or unreadable token does not prove that
persisted settings, app permissions, or files disappeared. AUA retains their undos and reports
that deliberate recovery is required, without replaying onto an unproven boot. This also applies
to target-facing services such as helper removal. Host-only residue can still be cleaned.
Missing backup files and unreadable ledgers are failures, never successful empty cleanup.

### Explicit recovery when the original boot or configuration is gone

`aua teardown status` reports each blocked target or corrupt ledger file separately. A malformed
file does not hide the remaining targets or stop their cleanup. Repair an unreadable file from a
backup at the reported `ledger_path`; AUA cannot infer its original target or undo arguments.

If the original boot still exists, restore its configuration (including referenced environment
values and `.platform-options-hmac-key`) and use `aua teardown run --serial-target <id>`.
`--force` only overrides a live-holder check; it never bypasses boot or configuration identity.
Credential rotation while undos are pending therefore blocks new mutations until the original
value is restored and the pending changes are undone.

If an operator has established that a disposable boot is permanently gone, they can explicitly
abandon individual recovery records after finishing/releasing its lease:

```sh
aua --platform example-os teardown discard --serial-target retired-target \
  --key screen_recording --reason 'Disposable boot was retired' --confirmed
```

This command never loads an adapter or connects to a target, so it also works for an uninstalled
plugin or lost configuration key. It archives the named entries, reason, and requester pid under
the ledger directory's private `discarded/` directory **before** removing them from automatic
replay. The result names that recoverable archive and explicitly says `restored: false`.
Other undo keys remain pending. A live lease refuses discard; confirmation is not authority to
destroy another worker's recovery evidence. This is a deliberate operator decision, not an
automatic retry strategy or proof that device state was restored. MCP exposes the same Engine
operations as `teardown_status`, `teardown_run`, and `teardown_discard`.

MCP is a trusted-client boundary: `confirmed: true` is an explicit request, not independent proof
of human approval. Restrict recovery tools in the MCP client's allow-list or permission policy;
an agent must obtain user authorization before forced replay or discard. With leasing disabled,
the operator must additionally establish that no live worker still relies on the selected undos.

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
`aua teardown run` forces it by hand. The dashboard's **Unlease** button on the Lease chip is the
same escape hatch for a human: it runs the identical clean-then-release pair under the same device
lock, so breaking a wedged lease from the browser can never hand the next agent a still-mutated
device.

### 4. The idle emulator watchdog, lease-gated

The existing idle watchdog stopped an aua-started headless emulator after wall-clock idleness alone.
That is a weak signal — an agent can legitimately sit in a 120s wait — so the timeout had to stay
long (900s) to be safe. It now also requires **no live lease**, which is a strong signal precisely
because a dead owner's lease expires instantly. That is what makes
`teardown.emulator_idle_stop_s: 120` a reasonable thing to configure; the default is 1200 seconds
so manual use of a windowed emulator is not mistaken for AUA activity. It reaps the device's ledger before stopping it, because the host-side
residue outlives the emulator.

An emulator AUA did not start is never touched.

#### Warm handoff between goal sessions

Back-to-back goal sessions do not turn emulator boot into per-scenario setup. The normal
session boundary restores every owned mutation and releases the process-bound lease, but leaves a
healthy AUA-started emulator online and unleased. A later `session start` can then select that warm
target; its fresh app install provides scenario isolation without requiring an emulator reboot.

The lease-gated idle watchdog above owns retirement. If no next session arrives before
`teardown.emulator_idle_stop_s`, it reaps the ledger and stops the emulator. A live lease always
blocks retirement, and AUA never retires an emulator it did not start. Agents do not request, keep,
or stop warm targets themselves.

`session finish` reports this as `owned_emulator_handoff` with the serial and configured idle
timeout, after `lease_release` succeeds. A failed restore retains the lease so the same process can
retry safely; it does not publish a warm target with dirty state. A failed `session start` still
stops only the exact emulator instance that bootstrap created, and explicit emulator-stop or MCP
process-exit cleanup remain immediate ownership boundaries.

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
  emulator_idle_stop_s: 1200.0 # stop an aua-started emulator idle AND unleased this long (0=off)
```
