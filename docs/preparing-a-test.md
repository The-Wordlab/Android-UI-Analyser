# Preparing a test with the agent that wrote the feature

A goal is not a test.

> "The hub badge shows once on first open."

To prove that, AUA needs a build, a way past sign-in, a definition of *first open* it can actually
establish on a device, a decision about whether the backend is in scope, and a statement of what
should be on screen. The agent that just implemented the feature has all of it. AUA has none of it.

`aua prepare` is the conversation that closes that gap, and the scenario it leaves behind so the
conversation happens **once**.

Nothing here changes how AUA works without it. `aua session start` is untouched, and a prepared
scenario is an ordinary contract file you can run by hand.

---

## The shape of it

```
agent → aua prepare start --goal "badge shows once on first open" --app com.example.app

aua   → knows already: how this app signs in, where its build is   (from the app map)
        needs to be told:
          1. what "first open" is, in terms the app stores
          2. any feature flags that must be on for it to exist at all
          3. how to reach that state  →  AUA suggests: datastore
                                         (reversible, ~1s; proves nothing about how the key
                                          gets its first value on a genuinely new install)
          4. a saved flow that gets past sign-in, if there is one
          5. UI only, or through the backend?
          6. what must be on screen when it works?
          7. and what must be on screen the second time?

agent → reads its own source, answers

aua   → writes the contract, saves the scenario, remembers the durable facts,
        and hands back the run command

next time → aua prepare run badge-shows-once --app com.example.app
```

Over MCP the same five steps are `prepare_start`, `prepare_answer`, `prepare_show`,
`prepare_list`, `prepare_run`. The first four need no device.

---

## Answering

Answers arrive as `key=value`:

```bash
aua prepare answer prep-4f2a91c0b3de --app com.example.app \
  --answer build=/path/to/app-debug.apk \
  --answer signin='tap `Continue as guest` on the welcome screen' \
  --answer precondition='the DataStore key hubBadgeSeen is absent' \
  --answer seeding=datastore \
  --answer scope=ui \
  --answer success='rid:hubBadge,text:New' \
  --answer repeat='!rid:hubBadge'
```

You can answer in as many calls as you like; the interview is stored beside the app map and
survives between processes, which is the point — the agent usually has to go and read code between
the question and the answer.

### `success` and `repeat` are predicates, not prose

These two become the contract, so they are written in the predicate grammar AUA already uses for
`until:` — comma-separated, `!` for "must not be there", a bare word meaning visible text:

| You write | It asserts |
|---|---|
| `rid:hubBadge` | an element with that resource id exists |
| `text:New` | visible text `New` exists |
| `desc:Badge` | that content description exists |
| `New` | visible text `New` exists |
| `!rid:hubBadge` | that element is **absent** |
| `desc:"Create, New"` | a value containing a comma — quote it, or the comma splits it in two |

**AUA will not translate a sentence into an assertion.** A generated oracle that looks right and
asserts the wrong thing is worse than no contract, so the translation stays mechanical and the
`provenance` in the result shows you every checkpoint next to the answer it came from. Read it.

### `repeat` is what makes "once" testable

A goal that says *once*, *first run*, *one-shot*, or *not shown again* raises a second question,
and produces a second checkpoint. Without it a passing run only proves the thing happened at all.

---

## Seeding: AUA suggests, you decide

Every way of reaching a pre-condition fakes something. Which substitution is acceptable depends on
what the test is for, which is your call and not AUA's — so the recommendation is a visible score
you can disagree with, and each option carries what it does **not** prove.

| Strategy | Reversible | Cost | Fakes | Proves nothing about |
|---|---|---|---|---|
| `datastore` | yes | ~1s | local state | how the key gets its first value on a new install |
| `database` | yes | ~3s | local state | the migration or writer that produces that row |
| `flags` | yes | ~5s | the code branch | the default branch real users get |
| `mock` | yes | ~10s | the backend | whether the real backend sends that shape |
| `reinstall` | **no** | ~45s | nothing | — it is the real thing, at the price of all app state |
| `ui` | yes | ~60s | nothing | — but a once-only state usually cannot be re-entered |

Choosing `scope=e2e` pushes `mock` down the list: faking the backend removes the thing an
end-to-end run is for. Only `seeding=reinstall` ever wipes the app.

---

## What gets remembered, and what does not

Facts about the **app** are written to the app map under `prepare:<question>`, so the next
interview reuses them instead of asking: `build`, `signin`, `precondition`, `seeding`. The reuse
is stated in `reused_from_memory` with the knowledge id, never applied silently — a stale sign-in
recipe should be correctable before the run, not discovered during it.

Decisions about **this test** are not remembered: `scope`, `success`, `repeat`. Remembering those
would quietly answer a question the next agent has every right to answer differently.

**Only a previous answer may skip a question.** Merely *related* knowledge — the kind
`aua knowledge list` returns — is attached to the question as `related_knowledge` and shown to the
agent, never substituted for an answer. Relevance and answering are different relations, and
conflating them does not add a bad answer, it removes the question: found the first time this ran
against a real app, where a note about forced dark theme stood in as the answer to *where is the
build*, and the run would have started with no APK.

`--no-remember` keeps everything out of the map.

---

## Running it

```bash
aua prepare run badge-shows-once --app com.example.app
```

Two things can happen, and the result always says which:

**`driven_by: "aua"`** — a controller model is configured and its key is set, so AUA runs the
whole loop itself: install, seed, drive, judge against the contract, record. The calling agent
pays for one question and one answer instead of one round trip per tap. That is the entire reason
to turn it on.

**`driven_by: "caller"`** — no controller, so the same contract comes back as the ordered commands
to run yourself, with `why` naming what was missing. Not a degraded mode; it is how AUA has always
worked.

Turn the controller on in config:

```yaml
controller:
  enabled: true
  api_key_env: OPEN_ROUTER_API_KEY      # must actually be set, or it stays unavailable
  model: or-deepseek-v4-flash-0731-low-open
  judge_model: or-deepseek-v4p1-flash-low-open
  judge_fallbacks: [or-gemma4-26b-thinking]
  cost_limit_usd: 0.15                  # the budget is the stop, not max_tokens
  record: true
```

A controller that dies before it judges returns `blocked`, never `failed` — an infrastructure
problem is not a product verdict.

---

## The evidence comes back with the verdict

Every run returns an `evidence` block: screenshots, video, the written report, the raw call log,
grouped by kind with sizes and paths, so a caller can publish the bundle without re-walking the
directory and guessing.

```json
"evidence": {
  "counts": {"image": 12, "video": 1, "text": 2, "data": 3},
  "videos": [{"name": "journey.mp4", "bytes": 4201233, ...}],
  "report": ".../verdict.md",
  "review_before_publishing": ["aua/calls.jsonl", "logcat.txt"]
},
"publishing": "Read the files in evidence.review_before_publishing before sharing this bundle:
               device and network records can carry tokens, account addresses and identifiers.
               AUA has not read them for you."
```

Device and network records are **flagged, not dropped** — a disputed verdict is exactly when you
want the raw calls. But nothing in that list should reach a shared link before a person has read
it, and AUA does not pretend to have done that for you.

---

## Commands

| Command | What it does |
|---|---|
| `aua prepare start --goal <claim> --app <package>` | Open an interview; returns only what AUA cannot know |
| `aua prepare answer <id> --app <pkg> --answer k=v ...` | Answer; the last answer writes the scenario |
| `aua prepare show <id> --app <pkg>` | The interview so far, unchanged |
| `aua prepare list --app <pkg>` | Interviews in flight and scenarios already prepared |
| `aua prepare discard <id> --app <pkg>` | Drop an unfinished interview; saved scenarios are untouched |
| `aua prepare run <scenario> --app <pkg>` | Run it, driven by AUA or by you |

## Two answers that decide whether a run works at all

`flags` — a flag-gated surface is simply **absent** without its flag, and an absent surface looks
exactly like a broken one. `--answer flags='myFeatureExperiment=a, otherExperiment=b'`; they are
applied and read back before the run.

`setup_flow` — sign-in and onboarding are long, well known, and identical every run. Point at a
saved AUA flow and it is replayed in seconds instead of rediscovered by the model every time:
`--answer setup_flow=flows/common/enter-as-guest.yaml`.

Both are optional, both are remembered per app, and `none` is a real answer to either.

**A setup flow is the fast path, not the oracle.** Flows go stale — a screen is renamed, an arrival
marker moves, a cold start outlasts the wait ceiling — and when one does, the app is still running
and still on a screen. So a setup flow that runs out of wait budget is re-issued from the step it
stopped on, and one that diverges for any other reason hands the controller the screen it reached,
with a note saying which part of the precondition is still owed. The adaptation comes back on the
handback as `adapted`, and it is in the verdict: a run that worked around a stale flow did not
reach its answer the same way as one that did not, and the agent deciding whether to trust the
verdict has to see that. What still ends a run before the model is spent: a flag that would not
apply, and a **prelaunch** flow, which sets up state nothing downstream can observe.
