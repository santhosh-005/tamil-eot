# `--label-policy` — settled

`llm` (default) vs `core-safe` (keep the pipeline's label on `change`).
Differ on **420 of 18,485 rows** — 2.27% of the corpus,
all in one direction: pipeline `complete`, LLM `incomplete`.

**Decision: `llm`.** Weak evidence for it, no evidence against, and
switching would re-label the frozen test set.

## 1. On the rows the policy decides, the human backs the LLM

Of the 197 blind-listened clips, **6** are `change` rows where the two
witnesses disagree — the only direct evidence there is.

| clip | pipeline | LLM | human |
|---|---|---|---|
| `007.wav` | complete | incomplete | **complete** |
| `080.wav` | complete | incomplete | **incomplete** |
| `090.wav` | complete | incomplete | **incomplete** |
| `105.wav` | complete | incomplete | **incomplete** |
| `109.wav` | complete | incomplete | **complete** |
| `175.wav` | complete | incomplete | **incomplete** |

**4–2 for the LLM.** On 6 clips that settles nothing on its
own — it is the direction, not a result.

## 2. What moves

| split | affected | of | share |
|---|---|---|---|
| train | 303 | 11,992 | 2.53% |
| dev | 47 | 2,325 | 2.02% |
| test | 70 | 4,168 | 1.68% |

**70 of them are in the sealed test split.** That is the argument that
actually decides this: `core-safe` does not re-weigh the training data, it
re-labels the benchmark. Every number in `RESULTS.md` would then be quoted
against a different ground truth than the one it was measured on.

## 3. And it would buy nothing

Shipped int8 models, scored on the affected test clips at their default
threshold. Change in overall test accuracy if the labels flipped:

| | Δ test accuracy |
|---|---|
| tiny int8 | -0.10 |
| base int8 | +0.29 |

Opposite signs, both far inside the **0.87-point** run-to-run spread of §5b.
There is no measurable difference between the two policies, so the tie
breaks on benchmark integrity.

## Still reversible

Every row carries `label_pipeline`, `llm_verdict` and `dispute`. Switching
is one flag on `11_label.py` plus a repack — but it invalidates
the published numbers, so it is a decision to re-measure, not a config
change.

Regenerate with `python pipeline/20_label_policy.py`.
