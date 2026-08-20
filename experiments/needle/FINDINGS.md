# Needle 2 for AUA — measured findings, and why it was dropped

Date: 2026-08-18. Model: `cactus-needle` 2.0.6 (Needle 2, 45M params, 14 MB, Apache-2.0),
base weights, **no fine-tuning**. Host: macOS arm64, CPU, Python 3.11.

## Verdict

**Base Needle 2 cannot do any decision AUA needs.** It works on the narrow, common
API-shaped patterns in its training data and returns position 1 — or abstains — on everything
else. Not recommended for AUA without the same training pipeline FunctionGemma already has, and
fine-tuning would cost the calibrated confidence head that made it attractive in the first place.

Dropped. Do not re-attempt with base weights.

## What works

Load 2.78 s. Inference **0.05–0.23 s** per call. Grammar-constrained decoding is flawless: across
every probe it never emitted a value outside the declared schema. The confidence head is honest —
it reported 0.001–0.04 on exactly the cases it got wrong.

Its trained shape works well:

| Input | Output |
| --- | --- |
| `set an alarm for 7 to catch the train` | `{hour: 7, label: "catch the train"}` |
| `wake me at 6 for the gym` | `{hour: 6, label: "gym"}` |
| `alarm at 22 called laundry` | `{hour: 22, label: "laundry"}` |

3/3. So the model, the install, and the prompt shape are all sound. Everything below is a real
capability result, not a harness bug.

## What does not work

| Task | Result |
| --- | --- |
| goal → known screen name (enum of 5) | **1/5** — returned the first enum value every time |
| which visible string is this screen's title (title listed first) | 5/5 |
| **same task, title listed last** | **0/7** |
| **same task, every rotation of each list** | **0/7 order-invariant** |
| free-text destination extraction from a goal sentence (no enum) | **0/7** — empty every time |
| same, reworded | **0/7** |

## The finding that matters

The title-extraction result went **5/5 → 0/7** purely by moving the correct answer from the front
of the candidate list to the back. Rotating one screen's five labels produced five different
answers. The model is positional, not semantic: with an enum it picks index 0, without one it
abstains.

That is the same failure FunctionGemma v1–v8 kept hitting — aggregate accuracy that evaporates
under permutation. This experiment reached the verdict in about twenty minutes because the
permutation test was run *before* believing the first number, rather than after a training cycle.

**Reusable rule: permute candidate order before believing any selection result.** A single
ordering tells you nothing. The FunctionGemma pipeline already encodes this as
candidate-order/ID permutation tests; apply it to any future selector on day one.

## Why fine-tuning is not the obvious next step

- Tuning **disables the calibrated confidence head** (per Cactus's own fine-tuning docs), which was
  the main reason a near-zero-risk deployment looked safe.
- The 256-token sliding window rules out anything needing navigation history.
- It needs the same curriculum/eval/permutation machinery `experiments/functiongemma` already has
  after eight rounds — at 45M rather than 270M, on tasks 270M has not yet won.
- It adds a JAX dependency (jax, jaxlib, flax, optax, sentencepiece) to the tree.

## What the session produced anyway

The work done to make Needle *possible* was worth more than Needle:

- **`rollback` was broken.** It validated a restored snapshot absolutely and could refuse because
  of damage that snapshot already carried. A synthetic damaged-map regression now proves inherited
  errors are carried onto the event the way `apply` carries them.
- **`answer_many`.** A batch of naming answers is now one deep copy, one validation pass, one
  snapshot pair, and one rollback id.
- Both are model-agnostic. An agent can drain the backlog through them today.

## Reproducing

```bash
uv pip install cactus-needle           # pure-python wheel, pulls JAX
```

Then, for any candidate-selection probe, run **every rotation** of the candidate list and require
the same answer from all of them. Report order-invariance, never single-ordering accuracy.

---

## Postscript: better names do not make fuzzy goals resolve

`_find_targets` matches lexically over names, aliases, and anchors, so a rename only helps a caller
who already uses nearby vocabulary. A synthetic follow-up confirms that readable names improve map
legibility but do not supply missing synonyms.

Anyone evaluating this should write goal wording independently of the names under test, just as
candidate order must be permuted before believing a selection result. Raw goals and learned names
from user app maps must remain local.
