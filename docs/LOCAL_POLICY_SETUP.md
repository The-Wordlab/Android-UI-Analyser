# Turning on the local policy (and autopilot)

The local policy lets a small model on your Mac pick the next tap during a goal session, so a
parent agent stops paying tokens and round-trips for every step. It is **off in shipped defaults**
and stays off until you turn it on: `policy.enabled` is `false`, `policy.mode` is `off`, and while
that holds nothing is resolved, downloaded, or loaded, and no memory or compute is spent. You do
not need to do anything to keep it off.

The LoRA adapter now ships **inside the wheel**, so enabling it no longer means staging an adapter
or pinning hashes by hand. You still supply the base model yourself — see step 2.

**What it does not change:** the model never authors an action. AUA deterministically builds
complete, current-frame tap calls, removes the unsafe, unauthorized, destructive, stale, ambiguous,
and redundant ones, and shows the model only opaque integer IDs over a privacy-screened projection.
AUA keeps the authoritative call map and revalidates before executing. The model chooses among
choices AUA already considers safe; it cannot widen that set.

---

## 1. Install the optional extra (Apple silicon only)

Into the environment `aua` actually runs from. If that is the **global** install agents reach
through PATH, use the installer's opt-in — it puts the extra in the install target, so `uv`
records it in the tool receipt and the next `uv tool upgrade` keeps it:

```bash
./install.sh --with-policy            # small selector (mlx-lm)
./install.sh --with-policy=hybrid     # ... plus the MLX-VLM reviewer
```

For a source checkout you drive directly:

```bash
uv pip install -e ".[functiongemma]"
```

It is not in the default install on purpose: mlx alone is ~183 MB installed, the reviewer extra
additionally pulls opencv/fastapi/mlx-audio, and installing either is never sufficient anyway
because the lane also needs the base model from step 2. `aua session autopilot` is its own opt-in
at run time; `--with-policy` is the matching opt-in at install time.

The MLX runtime is Apple-silicon-only. On any other platform the provider reports unavailable and
the policy stays off.

`aua doctor` now reports this: a policy that is configured (`mode: shadow`/`advisory`) but whose
runtime is missing shows as a `FAIL` naming each provider and the exact install command, instead of
saying nothing at all.

## 2. Get the base model (~543 MiB, one time)

The adapter is a *modification of* Gemma, so the base is deliberately not in this repository, not in
the wheel, and never downloaded automatically. Review and accept the
[Gemma Terms](https://ai.google.dev/gemma/terms), then fetch the exact pinned conversion:

```bash
hf download mlx-community/functiongemma-270m-it-bf16 \
  --revision bb327a9ad61044e1496a2bee2365a6b6a6684c72 \
  --local-dir /absolute/path/to/functiongemma-270m-it-bf16
```

The revision matters: the bundled manifest pins each required base file and their aggregate digest,
and the provider refuses to run against anything else.

This mirror is not gated, so the download needs no Hugging Face token and no login — an agent can
run the command above unattended. That is a fact about the mirror, not permission: the weights are
Gemma-licensed either way, and agreeing to those terms is a decision for the person, not the agent.

**Steps 1 and 2 are both ordinary shell commands, so an agent can do them.** Step 1 assumes a clone
of this repo; someone who installed AUA as a tool rather than from source installs the extra the way
their installer expects instead.

## 3. Turn it on

In `~/.config/android-ui-analyser/config.yaml` (user-wide) or `.android-ui-analyser.yaml` (one
project):

```yaml
policy:
  enabled: false          # leave false — see below; autopilot does not need it
  mode: advisory          # off | shadow | advisory
  chain: [functiongemma]
  max_candidates: 4

models:
  functiongemma:
    model_path: /absolute/path/to/functiongemma-270m-it-bf16
    adapter_path: null    # null selects the packaged v10 LoRA — this is what you want
    max_tokens: 24
```

**Two switches, and they do different jobs.** `mode` says what the policy may do; `enabled` says
whether *ordinary* calls pay for it.

* `mode: advisory` is what `aua session autopilot` requires. Typing that command is the opt-in, so
  the command works with `enabled: false`.
* `enabled: true` additionally attaches a `policy_suggestion` to ordinary `analyze` and session
  progress calls. That runs the chain on **every** such call, which with the reviewer in play costs
  roughly twenty seconds each. Turn it on only if you want that passive advice, and expect the bill.
* `mode: off` refuses autopilot outright, whatever `enabled` says.

So the configuration above is the one to want: autopilot on request, no cost to anything else.
`shadow` runs the model but exposes no selection — a development setting for tracing decisions
without acting on them, with no end-user purpose.

Nothing about this is permanent. To switch the passive advice off for one command:

```bash
AUA_POLICY__ENABLED=false aua <command>
```

To refuse autopilot as well, set `AUA_POLICY__MODE=off`.

## 4. Check before touching a device

```bash
aua policy status
```

This validates config, dependencies, artifact hashes, and daemon compatibility **without** loading
the model or talking to Android. Expect `available: true`, `max_mode: advisory`, and
`source: bundled_manifest`.

## 5. Run it

```bash
aua session autopilot --max-steps 6 --max-duration-ms 30000
```

The warm daemon executes the guard-approved tap itself and repeats while another safe navigation tap
is available. It stops and hands back the fresh screen on model handoff, a stale or unknown outcome,
an unchanged frame, a repeated call, work it does not do (input, toggles, waits, proofs), or the
step/time limit. **It executes taps only.**

It also steers only toward the phase the session says the run is on. Autopilot reads the authored
navigation waypoints of the *active* goal phase; it follows a later phase's waypoint only once every
waypoint of the earlier phases is reached, and the step that does so records the crossing
(`active_phase`, `phase`, `crossed_phases` in the trace). A phase it cannot reach by tapping — a
proof-only checkpoint, an offline transition, a cleanup — stops the run with
`terminal_reason: phase_not_navigable` rather than picking some other screen. A waypoint nothing on
screen matched is reported in `skipped_waypoints`, never in `completed_waypoints`.

### When a provider's output is mostly unusable

The guard refuses model output it cannot resolve to an offered candidate ID, so an unusable model
never executes anything — but a model that fails four times in five is a broken provider, not a
fallback, and each of those attempts costs seconds. AUA counts valid and invalid selection attempts
per provider over a rolling window of the last 20:

* the rate rides along in the decision trace (`recent_invalid_rate`) and in `aua policy status`
  (`selection_health` per chain member);
* once the recent window is majority-invalid, the chain refuses that provider instead of consulting
  it, and says so in the trace (`status: provider_unusable`);
* if that leaves autopilot with nothing that can steer, the command fails **once**, before it
  touches the device, with `policy_autopilot_unusable` and the measured rate — instead of handing
  off on every step while reporting itself as working.

The window is in memory, so a restarted daemon re-measures from scratch. `mode: shadow` never
refuses a provider: shadow exists to measure one.

---

## Optional: add the reviewer that made the device runs work

A single 270M model picking alone is noticeably worse than the two-tier chain. `selective_hybrid`
asks the small model first and consults a larger local reviewer only when the small one has no
consensus or the choice is semantically ambiguous — so most steps stay fast:

```yaml
policy:
  enabled: true
  mode: advisory
  chain: [functiongemma, gemma4]
  strategy: selective_hybrid
  primary_reviews: 2        # the small model's own votes must agree
  reviewer_reviews: 3       # the reviewer's votes, under permuted IDs and order
  candidate_scope: safe_visible
  max_candidates: 4

models:
  gemma4:
    model_path: /absolute/path/to/gemma-4-e4b-it-4bit
    max_tokens: 512
    max_mode: advisory      # required: advisory use is an explicit local choice
```

Both tiers get the identical screened projection. Votes are taken with candidate IDs and list order
permuted, and unanimity is required, so a model that is merely reacting to position or ID does not
reach a verdict. If both tiers decline, control returns to the calling agent — that is the intended
end state, not a failure.

## Switching models without editing the file

Profiles deep-merge over the base, so a profile lists only what differs:

```yaml
profiles:
  shadow:
    policy:
      mode: shadow
  qwen3:
    policy:
      chain: [qwen3, gemma4]
```

```bash
aua --profile shadow <command>        # or: AUA_PROFILE=shadow aua <command>
```

A `qwen3` provider also exists for a text-only Qwen3 LoRA. Its adapter is **not** bundled; point
`models.qwen3.model_path`/`adapter_path` at your own local directories and set
`max_mode: advisory`.

## Collecting traces while developing

```bash
AUA_POLICY_TRACE_DIR=/absolute/path/to/traces aua session autopilot ...
```

To collect continuously, export it from `~/.zshenv` rather than `~/.zshrc`. Zsh sources `.zshrc`
for interactive shells only, and the variable has to reach the shell that starts the **daemon** —
that is where autopilot decisions are made, and an agent or script usually starts it
non-interactively. Set in `.zshrc` it records nothing from those runs, and does so silently.

Confirm recording is live, including how many records exist:

```bash
aua policy status | jq .training_trace
```

Records the exact model-facing prompt and what happened next, as `decisions.jsonl`. Off unless that
variable names a directory — there is no config key, so configuration drift cannot switch it on, and
nothing reaches the journal, the dashboard, or telemetry. The recorded prompt is the same screened
projection the model reads; trusted call arguments and device identity are not in it. What the model
chose is stored as *state*, not as a label — its own choice is not an oracle.

---

## Using it: when it helps, when it cannot, and how to phrase the goal

Everything in this section is expressed with synthetic examples that preserve the failure modes.

### First: confirm the model can actually run

The single most common failure is not a bad decision — it is the model never running at all. The
chain then hands off every step, which from the outside looks like "the policy is slow and useless"
rather than "a dependency is missing":

```
functiongemma  unavailable  optional dependency missing; install android-ui-analyser[functiongemma]
gemma4         unavailable  optional dependency missing; install android-ui-analyser[hybrid-policy]
```

So always start here:

```bash
aua policy status | jq '{ready, providers: [.providers[] | {provider, available, reason}]}'
```

`ready: true` and `available: true` for every provider, or nothing below matters.

**The trap:** if `aua` is a `uv tool` install, the extras must be part of the tool's own
requirements. Installing them into the tool's environment by hand works until the next
`uv tool upgrade`, which rebuilds that environment from the receipt and silently drops them. Check
`~/.local/share/uv/tools/android-ui-analyser/uv-receipt.toml` — if it does not name the extras,
re-run `./install.sh --with-policy` (or `=hybrid`), which installs them as extras *of the install
target* so the receipt records them:

```toml
requirements = [
    { name = "android-ui-analyser", extras = ["functiongemma"], editable = "/path/to/clone" },
]
```

`aua doctor` reports the same fault without the archaeology — it names each configured provider,
whether it can run here, and the command that fixes it.

### What this lane can and cannot do

It executes **safe navigation taps and nothing else**. It cannot type, toggle, scroll, wait, or
assert. That means it cannot get you through a login form or a text field, and there is no
configuration that changes this — it is the boundary the guard enforces.

So do not start autopilot on a login screen. Get the app to the first screen where the remaining
work is tapping, then hand over.

### Phrase the goal with the words that are on the screen

The goal is not only read by the model. A deterministic check compares the goal's words against
each candidate's label, and a choice sharing **no** word with the goal is refused outright, by the
reviewer as well as the small model. This is deliberate — it is what stops a confident tap on an
unrelated control when the target is absent — but it has a consequence worth planning around:

> The policy cannot cross a screen whose controls share no vocabulary with the goal.

In a synthetic catalog run, the goal was `catalog listing browsable products stop there without
changing anything`. On the login screen the only controls were *Continue with provider*, *Use demo
account*, *Continue as guest*; on the next screen, *Allow* and *Ask me later*. None of them shares a
word with that goal, so every one was refused. The three steps that did succeed were the
ones where a control was literally labelled **Apps**.

Practical rules:

* Name the control you expect to tap, in the words the screen uses. `Open Catalog` beats
  `catalog listing browsable products stop there without changing anything`.
* Keep it short. Extra words such as *stop there without changing anything* add nothing for the
  model and dilute the overlap check.
* One destination per run. If the route crosses a permissions dialog or a login, drive that part
  yourself and start autopilot after it.

### Per run, not for the whole session

Autopilot is a bounded, explicit command, not a mode you switch on for a session:

```bash
aua session autopilot --max-steps 6 --max-duration-ms 30000
```

Each invocation runs until it reaches a limit or hits anything it does not do, then returns the
fresh screen and control. Call it again for the next stretch. Nothing is remembered between calls
beyond the session itself, and there is no "leave it driving" setting — by design.

The model is loaded for the duration of the command and nothing else pays for it, which is why
`enabled` can stay false.

### Reading a run that did nothing

`aua policy status | jq .training_trace` shows whether decisions are being recorded and how many
exist. With recording on, each decision line carries every provider's attempt and its reason, which
is what distinguishes the three failures that look identical from the outside:

| What the trace says | What it means |
|---|---|
| `unavailable: optional dependency missing` | the model never ran — fix the install |
| `no_consensus` | the model's own votes disagreed; the next tier was asked |
| `rejected_semantic` + `no_goal_overlap` | nothing on screen matched the goal's words — rephrase, or the target is genuinely absent |

## What to actually expect

Honest numbers, from a probe written independently of the training data (150 jobs):

| | accuracy | refusal | latency | peak memory |
|---|---|---|---|---|
| bundled v10 (270M) | 0.600 best checkpoint, 0.471 mean | 18/38 | ~180 ms/decision | ~1.9 GB |
| Qwen3-1.7B (not bundled) | 0.667 best, 0.598 mean | 4/38 | ~419 ms/decision | ~3.9 GB |

On a device the bundled adapter completed 5/5 navigations and 2/2 refusals with zero wrong taps.

Read that as "useful for a bounded lane", not "solved":

* **It is not promoted.** One seed, no live gate, and refusal swings between 0 and 18 across
  checkpoints — the 18/38 is a high-water mark of a noisy process, not a converged property.
* **Refusal is not load-bearing.** No model in this line reliably declines. The deterministic guard
  is what keeps a wrong choice from becoming a wrong action, which is why the model is only ever
  offered candidates that are already safe.
* **Expect handoffs.** Roughly a third to a half of decisions come back to the calling agent. That
  is the design working: the lane is a cost optimisation for the easy steps, not an autonomous agent.

Why this matters when reading any score: an in-house probe that shared phrasing with the training
generator reported 6/6 on a refusal capability that independent measurement put at 0/144. If you
evaluate a new adapter, write the probe separately from the data.
