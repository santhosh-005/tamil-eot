# 07 — Choosing the decision threshold ⚠️

**0.5 is inherited from the training loop, not chosen. Neither model peaks
there.** Tuning it away turned out to be harder than it looks.

---

## ❌ Tuning for accuracy on `dev` — did not transfer

Pick the argmax on `dev`, then measure once on `test`:

| | threshold | dev acc | **test acc** | test FP/N |
|---|---|---|---|---|
| tiny, inherited | 0.50 | 84.56% | **83.35%** | 7.73% |
| tiny, dev-argmax | 0.34 | 84.80% | 83.81% | 9.57% |
| base, inherited | 0.50 | 87.21% | **86.23%** | 8.95% |
| base, dev-argmax | 0.24 | 87.50% | **85.10%** | 11.95% |

**base's tuned threshold lost 1.13 points.** +0.29 on dev, −1.13 on test — a
textbook fit to the selection set.

Why: the accuracy-vs-threshold curve is **flat near its top** and dev is 2,325
clips, so the argmax is mostly noise. The test sweep's best says 0.65; the dev
sweep says 0.24. **That disagreement is the same fact stated twice.**

## ✅ Targeting a *rate* transfers; targeting an argmax does not

Tightest threshold holding **dev FPR ≤ 10%** — the agent interrupts on at most
one pause in ten:

| | threshold | test acc | test FP/N | vs 0.5 |
|---|---|---|---|---|
| tiny `polite` | 0.75 | 80.64% | **5.01%** | −2.71 acc, −2.72 FP/N |
| base `polite` | 0.92 | 82.41% | **4.51%** | −3.82 acc, −4.44 FP/N |

**FP/N roughly halves** for ~3 accuracy points — better than every language in
Smart Turn's published table, English included.

The 10% budget itself slips to ~12–14% on test. **The direction transfers, the
level does not.**

## What ships

| operating point | tiny | base | |
|---|---|---|---|
| **`inherited`** | **0.50** | **0.50** | **default — every published number** |
| `polite` | 0.75 | 0.92 | halves FP/N, costs ~3 points |
| `balanced` | 0.34 | 0.24 | dev-best accuracy — see above, did not transfer |

`smart-turn-v3` gets only `inherited`: there is no dev split of its data to pick
one on, and inventing an operating point for someone else's model is guessing
with a number that decides interruptions.

## The ceilings, for scale

Fitted **on test** — maxima of a sweep, not settings:

| threshold fitted on test | tiny | base |
|---|---|---|
| for 10% FPR | 0.88 | 0.94 |
| accuracy there | 75.65% | 80.97% |
| best-accuracy threshold | 0.30 | 0.65 |

Held to the operating point a voice agent actually runs, the gap between the
two models **more than doubles**.

> Quote the curve, not the row — and quote the curve you did **not** select on.

## The two conventions that break comparisons

**FP/N ≠ FP/(FP+TN).** Smart Turn publishes the first; they differ by ~3×.
On this test set 27.4% standard is 10.1% theirs. Mixing them makes a competitive
model look broken. Every table here states which it uses.

---

`reports/operating_point.md`, `reports/finetune_results_*.md` ·
`python pipeline/18_operating_point.py`
