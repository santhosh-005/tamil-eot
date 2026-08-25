# 04 — What actually moves the number

Four levers measured. One worked.

| lever | ΔAUC |
|---|---|
| +20% training rows (funnel harvest) | +0.003 |
| learning-rate retune | +0.003 |
| 50% → 100% of the data | +0.001 |
| **whisper-tiny → whisper-base** | **+0.018** |

Three levers, ~0.003 each. One lever, 6× that. **Weeks spent on the wrong axis.**

---

## ✅ Encoder capacity — the win

One string change: `openai/whisper-tiny` → `openai/whisper-base`.

**+2.88 accuracy, +0.018 AUC.** 8.0M → 20.3M params.

Fixed the buckets where tiny lost to the majority-class policy:

| bucket | tiny | base | |
|---|---|---|---|
| `change` | −7.42 | −2.05 | still under |
| `change_midseg` | −1.34 | **+0.45** | **first time above** |
| `hold_intra` | +31.59 | **+33.42** | the real EOT problem |
| `hold_inter` | +31.88 | **+35.25** | |
| `disputed` | −11.76 | −6.17 | still under |

## ⚠️ And then the constraint moved back to data

Two diagnostics, both run *at base*, agreeing:

| diagnostic | tiny (8M) | base (20.3M) |
|---|---|---|
| learning curve at 100% | **flat** (0.909 → 0.910) | **rising** (0.922 → 0.931) |
| train − dev accuracy gap | small | **+11.81** (98.99 vs 87.18) |
| verdict | capacity-limited | **data-limited / overfitting** |

The honest statement is sequential, not absolute: *tiny was capacity-bound;
base is data-bound.* A 20M-param encoder memorises 11,992 clips (train AUC
0.9892) while dev sits at 0.9234.

**Neither diagnostic transfers across an architecture change.** Re-run the
train−dev gap after any capacity change — one forward pass, ~1 minute, and it
is what caught this reversal.

## ⚠️ Learning rate — swept twice, conclusion flipped

Upstream trains batch 384 @ 5e-5. This uses batch 32.

| sweep | train rows | winner |
|---|---|---|
| #1 | 10,005 | **1.4e-5** (sqrt-scaled) |
| #2 | 11,992 | **5e-5** (upstream, unscaled) |

**Sqrt-scaling was never a law.** It was compensating for a training set small
enough that 5e-5 overfitted. With 20% more data it does not.

**Lesson: the right LR depends on dataset size.** Both shipped models are 5e-5.

## ⚠️ Learning curve — architecture-specific, not wrong

| capacity | 25% → 50% → 100% (dev `hold_intra` AUC) | |
|---|---|---|
| tiny, 10,005 rows | 0.886 → 0.891 → 0.920 | rising |
| tiny, 11,992 rows | 0.865 → 0.909 → 0.910 | **flat** |
| base, 11,992 rows | 0.909 → 0.922 → 0.931 | **rising** |

Each correct *for its capacity*. Reading any one as a permanent fact is the
error. Individual points carry ±2.3 pts of sampling error on the dev subset —
read the trend, never one row.

**The control that made it mean something:** epochs were fixed, so the 25% run
took ¼ the optimiser steps. Re-run at 25% for 4× the epochs — same 1,872 steps —
returned **−0.17 points**. Four times the compute on the same data bought
nothing; four times the data at fixed compute bought +3.32. **Data, not compute.**

## ⚠️ Run-to-run spread — the seed is not enough

Three base runs, identical config, `SEED=0`:

**86.23% / 85.63% / 85.36%** — spread **0.87 points**, mean 85.74%.

`cudnn.deterministic` only constrains cuDNN ops. The encoder is attention, and
`transformers` routes through SDPA, whose flash/mem-efficient **backward uses
atomics**. Non-associative float addition → divergent trajectories over 2,244
steps.

Amplified by **checkpoint selection**: run 1 peaked at epoch 3, run 2 at epoch 4.
Different snapshots of a noisy curve, not different model quality.

Fix (expensive, untested): `use_deterministic_algorithms(True)` +
`CUBLAS_WORKSPACE_CONFIG=:4096:8` + disable flash/mem-efficient SDPA.

> **Treat anything under ~1 point as noise.** This is the single most important
> number in this repo for reading every other number in it.

## ❌ Checkpoint selection on dev AUC

Proposed as a cheap fix for the spread. Same encoder, lr and seed as the shipped
base; only the selection criterion differs.

| selected on | accuracy | AUC | FP/N | @10% FPR |
|---|---|---|---|---|
| **dev accuracy** (shipped) | **86.23%** | **0.922** | **8.95%** | **80.97%** |
| dev AUC | 85.63% | 0.917 | 10.20% | 75.19% |

Loses on accuracy, on AUC, and on **FP/N — the metric that decides whether the
agent talks over the user**. Its one win is FN/N, the cheap error.

**What this does *not* show:** that AUC-selection is unstable. 85.63% sits
inside the 85.36–86.23 band. One draw cannot measure a criterion's variance —
that needs N runs per criterion. "Threshold-free and far steadier" was asserted,
not measured. **Withdrawn.**

## ✅ Are the shipped files the models that were trained?

Two checks, because "we trained something good" and "the file we published is
that thing" are different claims.

| check | result |
|---|---|
| both shipped ONNX re-scored locally on the sealed test set | reproduces the Colab numbers exactly |
| `15_export_ckpt.py` re-exports the shipped base `.pt` → ONNX | **bit-identical** to the shipped graph, `max｜delta｜ 0.00e+00` |

So the scoring path used for every number here is the same one that produced
86.23%, and the export is not a re-training in disguise.

Export runs on **CPU** deliberately: the notebook's export cell is the last one
and the first casualty of a Colab usage limit. Three trained checkpoints came
off one session with no ONNX beside them.

---

`reports/lr_sweep.md`, `reports/learning_curve.md`,
`reports/finetune_results_{tiny,base,base_auc}.md` ·
`notebooks/train_smart_turn_tamil.ipynb` §5a–5c
