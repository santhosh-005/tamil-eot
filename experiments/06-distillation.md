# 06 — Distillation base → tiny ❌

**Question.** base is +2.42 points but 2.4× the size. Can tiny inherit some of
that without growing?

**Setup.** Teacher = the 86.23% base checkpoint. Student = tiny. `ALPHA=0.3`,
`T=2.0`. Identical seed, epochs, lr and data to the 83.35% tiny baseline — **the
loss function was the only variable**, so the delta is attributable.

---

## Result

| tiny | accuracy | AUC | FP/N | FN/N | @10% FPR |
|---|---|---|---|---|---|
| baseline (hard labels) | 83.35% | 0.904 | **7.73%** | 8.93% | 75.65% |
| distilled | 83.76% | 0.906 | 9.14% | **7.10%** | **77.52%** |
| delta | +0.41 | +0.002 | **+1.41 worse** | −1.83 | +1.87 |

**+0.41 is inside the ~0.9-point seed spread.** It closed **14%** of the teacher
gap against a textbook 30–60%. And FP/N — the metric that decides whether the
agent talks over the user — got *worse*.

## Where the gain came from, and it is the wrong place

| bucket | share | baseline | distilled | Δ |
|---|---|---|---|---|
| `change` (93% single-class) | 24.6% | 85.74% | 88.96% | **+3.22** |
| `hold_intra` — the real EOT problem | 58.0% | 81.90% | 81.45% | **−0.45** |
| `hold_inter` | 12.1% | 83.76% | 83.37% | −0.39 |
| `change_midseg` | 5.4% | 87.05% | 85.71% | −1.34 |

`+3.22 × 0.246 − 0.45 × 0.58 ≈ +0.41` — **the whole headline gain is `change`**,
the bucket where a speaker change already implies the turn ended.

**Every bucket that requires hearing prosody got worse.** The teacher
transferred its edge on structural cases and nothing on the hard ones — which
is what a teacher that memorised its training set (98.99%) has left to give.

## ⚠️ Do not pick the shipping tiny from this table

Both checkpoints were already selected on `dev`. Choosing the winner on `test`
is selecting on the benchmark. On what is measured, **baseline tiny keeps the
drop-in slot**.

---

`reports/finetune_results_distil.md` · the distillation notebook was an
experiment branch of `notebooks/train_smart_turn_tamil.ipynb` and is not shipped
