# Is data the bottleneck? — measured

> **2026-08-22: this curve is architecture-specific, and it was superseded.**
> Everything below was measured on **whisper-tiny** at 10,005 training rows.
> Re-run later at 11,992 rows, tiny's curve went **0.865 → 0.909 → 0.910** —
> flat, i.e. capacity-limited. Swapping to **whisper-base** then moved it to
> **0.909 → 0.922 → 0.931**, still climbing, and the train−dev gap opened to
> **+11.81 points** (98.99% train vs 87.18% dev): base *overfits*, so at base
> capacity data is the constraint again.
>
> Neither answer transfers across an architecture change. The cheap check is
> **§5a in the notebook** — train vs dev accuracy in one forward pass, ~1
> minute against this file's three training runs. It reproduced the full
> curve's verdict at base.
>
> Read this file as a record of the tiny-capacity measurement, not as a
> standing answer. Final numbers live in `RESULTS.md`.

The question behind every remaining decision: with test accuracy at 84.29%,
does the next point come from **collecting more data** or from **a bigger
encoder**? Those cost weeks and an afternoon respectively, so guessing is
expensive.

Trained on 25% / 50% / 100% of the 10,005-sample training set, stratified by
`(source, label)` so the mixture is identical at every size and only `n` varies.
Scored on **dev `hold_intra`** — 1,206 clips, majority baseline 55.31%.

`hold_intra` rather than overall accuracy because `change` and `change_midseg`
are 93% and 88% single-class: a speaker change all but implies the turn ended,
so 30% of the corpus is nearly free and would flatten any slope. `hold_intra` is
58% of the data, near-balanced, and is the actual EOT problem — same speaker,
mid-conversation, no structural cue.

## The curve

| fraction | n train | dev all | dev `hold_intra` | vs baseline | AUC (`hold_intra`) |
|---|---|---|---|---|---|
| 25% | 2,502 | 82.55% | 81.51% | +26.20 | 0.886 |
| 50% | 5,004 | 80.59% | 80.51% | +25.21 | 0.891 |
| 100% | 10,005 | **86.08%** | **84.66%** | **+29.35** | **0.920** |

| step | Δ `hold_intra` | 95% CI | |
|---|---|---|---|
| 25% → 50% | −1.00 | ±3.13 | noise |
| 50% → 100% | **+4.15** | ±3.02 | **significant** |
| 25% → 100% | +3.15 | ±2.99 | significant |

The dip at 50% is sampling error on n=1,206. The signal is AUC, which is
threshold-free and rises monotonically — **0.886 → 0.891 → 0.920** — with the
gain *accelerating*, +0.005 then +0.029. A model near its capacity ceiling shows
the opposite.

## The control that makes it mean something

Epochs were fixed at 6, so the 25% run took 468 optimiser steps against the full
run's 1,872. Some of the gain could have been compute rather than data. Re-run
at 25% for **24 epochs** — the same 1,872 steps:

| 25% subset | steps | dev all | dev `hold_intra` | AUC |
|---|---|---|---|---|
| 6 epochs | 468 | 82.55% | 81.51% | 0.886 |
| 24 epochs | 1,872 | 82.21% | 81.34% | 0.882 |

Four times the compute on the same 2,502 samples returned **−0.17 points**.
Training loss reached 0.0123 by epoch 23: the model memorised the subset and dev
never moved.

Holding compute constant at 1,872 steps and varying only the data:

| | train n | dev `hold_intra` | AUC |
|---|---|---|---|
| 25% × 24 epochs | 2,502 | 81.34% | 0.882 |
| 100% × 6 epochs | 10,005 | **84.66%** | **0.920** |
| | **4×** | **+3.32** | **+0.038** |

**Data, not compute. Collecting more is the highest-value move available.**

## What 90% costs

`hold_intra` is 57.9% of the test set, so overall accuracy is largely a function
of it. From the current 82.07% on test:

| `hold_intra` | overall | + `change` no longer losing to its baseline |
|---|---|---|
| 86% | 86.56% | 87.59% |
| 88% | 87.72% | 88.75% |
| **90%** | 88.88% | **89.91%** |

Extrapolating the one doubling actually measured (+4.15 per 2×):

| train size | if returns hold | if returns halve each doubling |
|---|---|---|
| 20k | 86.2 | 86.2 |
| 40k | **90.4** | 88.3 |
| 80k | 94.5 | 89.3 |

**~40k training samples — 4× the current set — puts 90% in range**, between
"just barely" and "comfortably" depending on how fast returns decay. A larger
encoder clears it in the pessimistic column too. The 722 label-less recordings
(~208 h) are the obvious source and 4× is a plausible yield.

## Caveats

- **Every run here used lr 5e-5, which [`lr_sweep.md`](lr_sweep.md) later showed
  is 3.5x too high for batch 32.** A correct learning rate lifts all four points
  together — the curve was re-run in a later session and reproduced AUC 0.920 at
  100% exactly — so the slope, and therefore the data-limited verdict, holds.
  The absolute values are all ~2 points low.

- Three runs of the same config gave test **82.51% / 82.10% / 84.29%**. No seed
  is set anywhere in the notebook, so init and shuffle differ every run. The
  spread is ~2 points — the same size as most effects discussed here. Set a seed
  before the next comparison.
- Every number above is a single run. The curve's shape is supported by the
  control; individual points are not.
- Scored on `dev`. `test` was touched once, at the end, as designed.
