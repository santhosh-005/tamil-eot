# Learning rate — 5e-5 was wrong for batch 32, and then it was right again

> **2026-08-22: this conclusion reversed once the training set grew.** Re-run
> on 11,992 train rows (up from 10,005 after the funnel relaxation), the same
> sweep picks **5e-5**, the value it had rejected:
>
> | lr | dev all | dev `hold_intra` | AUC | AUC(`hold_intra`) |
> |---|---|---|---|---|
> | **5.0e-05** | **84.52%** | **84.78%** | **0.9145** | **0.9139** |
> | 1.4e-05 | 83.96% | 83.73% | 0.9031 | 0.9057 |
> | 4.0e-06 | 82.97% | 81.56% | 0.8856 | 0.8803 |
>
> Square-root scaling was never a law — it was compensating for a training set
> small enough that 5e-5 overfitted. With 20% more data it does not, and the
> larger step wins by +1.05 points on `hold_intra` and +0.008 AUC(hi).
>
> **Acted on: both shipped models are now at 5e-5.** An earlier note here said
> the shipped ONNX was stuck at 1.4e-5 because `/content/lr_5e-05.pt` died with
> its session; `§5c` now persists `lr_*.pt` to Drive, so it survived. tiny at
> 5e-5 scores 83.35% (FP/N **7.73%**, better than 1.4e-5's 10.10% despite
> −0.22 accuracy) and base at 5e-5 scores 86.23%.
>
> This sweep's own 5e-5 checkpoint was later exported and scored on test:
> **85.63%, AUC 0.917** — see `RESULTS.md` §5b. It does *not* beat the shipped
> base, because the sweep selects on dev AUC and dev AUC is not FP/N.
>

> Read the original sweep below as *"the right LR depends on the dataset
> size"*, not as *"1.4e-5 is the right LR"*.

## Original sweep (10,005 train rows)


Upstream Smart Turn trains at **batch 384, lr 5e-5**. This project uses **batch
32** and inherited the same learning rate: 12x fewer samples per step at the
same step size. Nothing flagged it, and every result before this sweep — the
zero-shot comparison, the fine-tuned numbers, the whole learning curve — was
produced at a learning rate 3.5x too high.

Three values, six epochs each, **same seed** so init and shuffle are identical
and step size is the only variable. Checkpoints selected on dev AUC, which is
threshold-free; scored on dev `hold_intra` (n=1,206), the bucket that actually
carries the task.

## Result

| lr | scaling rule | dev all | dev `hold_intra` | AUC | AUC(`hold_intra`) |
|---|---|---|---|---|---|
| 5.0e-05 | upstream, unscaled | 82.06% | 82.01% | 0.9033 | 0.9034 |
| **1.4e-05** | **square-root** | **85.59%** | **84.25%** | **0.9113** | **0.9099** |
| 4.0e-06 | linear | 80.39% | 79.77% | 0.8878 | 0.8814 |

**+2.24 points on `hold_intra`, +0.0065 AUC** on dev — but see below: most of
that did not survive contact with the test set.

## What it was worth on `test`

Both checkpoints scored on the sealed split, same session, learning rate the
only difference:

| | 5e-5 | **1.4e-5** | delta |
|---|---|---|---|
| accuracy | 83.16% | **83.54%** | +0.38 |
| **FPR (std)** | 28.27% | **25.80%** | **-2.47** |
| ROC-AUC | 0.887 | **0.898** | +0.011 |
| `hold_intra` | 80.75% | 81.37% | +0.62 |
| `agree` | 81.73% | 83.16% | +1.43 |
| `disputed` | 85.51% | 84.17% | -1.34 |

Dev promised +2.24 on `hold_intra`; test delivered +0.62. The dev figure was
inflated by picking the checkpoint on the same split it was then scored on —
the ordinary cost of selection, and a reason to quote AUC, which transferred
honestly (+0.0065 dev, +0.011 test).

The bucket split shows what actually changed: `agree` +1.43 against `disputed`
-1.34, near-offsetting, with FPR down 2.47. The lower learning rate **reduced
the model's bias toward `complete`**. Accuracy barely moves because the disputed
bucket is 93.9% `complete` and penalises exactly that correction; FPR, which
decides whether a voice agent interrupts the user, improves materially. Same
accuracy, better model.

**Keep the change** — it is free, it converges cleanly, and it cuts FPR. But it
is a ~0.4-point accuracy lever, not a 2-point one, and 0.4 sits inside the
2.2-point run-to-run spread. Nothing here changes the conclusion that data is
the only lever with real headroom left.

## The per-epoch traces say more than the endpoints

| lr | dev accuracy by epoch | |
|---|---|---|
| 5.0e-05 | 75.20 → 83.24 → 82.06 → 82.94 → 83.48 → 83.14 | bounces; AUC peaks at ep2 (0.9033) then decays |
| **1.4e-05** | 80.88 → 82.79 → 84.75 → **85.59** → 84.90 → 84.75 | smooth climb, clean peak at ep3 |
| 4.0e-06 | 60.29 → 71.86 → 80.39 → 81.08 → 82.79 → 83.04 | still climbing — undertrained at 6 epochs |

Square-root scaling is the right rule here, as expected for an Adam-family
optimiser. 5e-5 is genuinely too hot: it reaches its best ranking at epoch 2 and
gets worse for the remaining four. Linear scaling to 4e-6 overcorrects — it has
not converged by epoch 6, and the cosine schedule has already decayed the step
size by then, so more epochs would not rescue it.

The instability at 5e-5 also explains the run-to-run spread that has been
contaminating every single-run comparison in this project: test accuracy across
four runs of nominally identical config came out **82.51% / 82.10% / 84.29% /
83.16%** — a 2.2-point range with no seed set anywhere.

## What it does and does not change

**Changes:** the notebook default is now `EPOCHS, LR = 6, 1.4e-5`, and the §4
prose that said "batch 32 and a few more epochs is the sane translation" now
says the learning rate has to come down too. That omission is the bug this
sweep found.

**Does not change:** the data-limited verdict. The learning curve was re-run in
the same session, still at 5e-5, and reproduced:

| fraction | AUC(`hold_intra`) first run | second run |
|---|---|---|
| 25% | 0.886 | 0.890 |
| 50% | 0.891 | 0.886 |
| 100% | **0.920** | **0.920** |

Identical at 100%, ±0.004 at the smaller fractions, with the 50% → 100% jump at
+0.029 and +0.034. A lower learning rate lifts all points together; it does not
flatten the slope. See [`learning_curve.md`](learning_curve.md).

**Shifts slightly:** `hold_intra` moves from ~82% to ~84.25% on dev from the LR
fix alone, so the gap to the ~90% needed for 90% overall narrows from ~8 points
to ~6. At the measured +3 to +4 per doubling of the training set, the target is
still roughly **40k samples**, marginally easier than before.

## Caveat

Every number here is a single run per learning rate, on n=1,206. The shared seed
makes the three comparable to each other, but the ±2.3-point sampling error
still applies to each in isolation — which is why the ranking is taken from AUC,
not accuracy. The ordering is consistent across both metrics and across the
epoch traces, so the conclusion is safe; the exact +2.24 is not.
